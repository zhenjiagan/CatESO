"""
ESMFold Structure Evaluator

Evaluates protein sequence structural quality (pLDDT, pTM, etc.).
Supports batch evaluation and asynchronous evaluation.
"""

import torch
import torch.utils.checkpoint
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import warnings

try:
    import esm
    from esm.esmfold.v1.categorical_mixture import categorical_lddt
    from openfold.utils.loss import compute_tm
    from openfold.data.data_transforms import make_atom14_masks
    ESMFOLD_AVAILABLE = True
except ImportError:
    ESMFOLD_AVAILABLE = False
    warnings.warn("ESMFold is not installed; structural evaluation will be unavailable. "
                  "Run: pip install fair-esm[esmfold]")


class ESMFoldEvaluator:
    """
    ESMFold evaluator.
    Uses ESMFold to predict protein structures and compute quality metrics.
    """

    def __init__(self,
                 device: str = 'cuda',
                 chunk_size: Optional[int] = None,
                 use_bf16: bool = True,
                 no_recycles: int = 3):
        """
        Initialize the ESMFold evaluator.

        The model is loaded automatically from the local cache via TORCH_HOME.

        Args:
            device:      Compute device.
            chunk_size:  Chunk size for long sequences (None = automatic).
            use_bf16:    Whether to use BF16 precision to reduce GPU memory usage.
            no_recycles: Number of folding trunk recycle steps
                         (0 = 1 pass, lowest memory; 3 = 4 passes, highest quality).
        """
        if not ESMFOLD_AVAILABLE:
            raise ImportError("ESMFold is not installed. Run: pip install fair-esm[esmfold]")

        self.device = device
        self.chunk_size = chunk_size
        self.use_bf16 = use_bf16

        print(f"[ESMFold] Loading model from TORCH_HOME cache")
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().to(device)

        if use_bf16 and torch.cuda.is_available():
            self.model = self.model.bfloat16()
            print(f"[ESMFold] Using BF16 precision")

        if chunk_size is not None:
            self.model.set_chunk_size(chunk_size)
            print(f"[ESMFold] chunk_size set to {chunk_size}")

        # Enable gradient checkpointing on trunk blocks to save ~1.8 GiB of activation memory
        self.model.trunk.cfg.cpu_grad_checkpoint = True
        print(f"[ESMFold] Trunk gradient checkpointing enabled (~1.8 GiB activation memory saved)")

        self.no_recycles = no_recycles
        print(f"[ESMFold] Trunk recycles: {no_recycles} ({no_recycles + 1} total passes)")
        print(f"[ESMFold] Model loaded on device: {device}")

    def get_esm_embedding_weight(self) -> torch.Tensor:
        """
        Return the token embedding weight matrix from the ESM-2 model inside ESMFold.

        Returns:
            embedding_weight: [vocab_size, embed_dim] embedding weight matrix.
        """
        return self.model.esm.embed_tokens.weight

    @staticmethod
    def _esm_layer_with_attn(layer, x):
        """Single ESM-2 layer forward (with attention), used for gradient checkpointing."""
        x, attn = layer(x, self_attn_padding_mask=None, need_head_weights=True)
        return x, attn

    def _esm_forward_from_embeddings(self, esm_model, embeddings):
        """
        Manual ESM-2 forward pass supporting soft embeddings
        (bypasses nn.Embedding and type checks in the standard forward).
        Uses gradient checkpointing to reduce memory usage.

        Args:
            esm_model:  self.model.esm
            embeddings: [B, T, E] soft token embeddings.

        Returns:
            dict with keys:
                'representations': dict mapping layer index → hidden state [B, T, E].
                'attentions':      attention weights tensor (or None if not requested).
        """
        x = embeddings

        hidden_representations = {}
        hidden_representations[0] = x  # layer 0 = input embeddings

        # (B, T, E) → (T, B, E): ESM-2 expects sequence-first layout internally
        x = x.transpose(0, 1)

        # Iterate over Transformer layers with gradient checkpointing.
        # When use_esm_attn_map=False, skip storing attention maps (~840 MB saved).
        need_attn = getattr(self.model.cfg, 'use_esm_attn_map', False)
        attentions = [] if need_attn else None
        for layer_idx, layer in enumerate(esm_model.layers):
            x, attn = torch.utils.checkpoint.checkpoint(
                self._esm_layer_with_attn, layer, x
            )
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_attn:
                attentions.append(attn.transpose(0, 1).unsqueeze(1))

        # Final LayerNorm
        x = esm_model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)
        hidden_representations[len(esm_model.layers)] = x

        # Concatenate attention weights: [B, n_layers, n_heads, T, T] (only when requested)
        attentions = torch.cat(attentions, dim=1) if need_attn else None

        return {
            "representations": hidden_representations,
            "attentions": attentions
        }

    def predict_structure_from_soft_embedding(self,
                                              soft_sequence: torch.Tensor,
                                              esm_alphabet) -> Dict:
        """
        Predict protein structure from a soft one-hot sequence with differentiable gradients.

        Fully reproduces the ESMFold.forward logic:
        1. Manual layer-wise ESM-2 forward to obtain all hidden representations.
        2. Weighted layer fusion via esm_s_combine.
        3. MLP projection (esm_s_mlp / esm_z_mlp).
        4. Add amino-acid type embedding (with STE for differentiability).
        5. Forward through the folding trunk.
        """
        device = soft_sequence.device
        vocab_size = soft_sequence.shape[1]
        dtype = self.model.esm_s_combine.dtype

        # 1. Get ESMFold's internal ESM-2 embedding weight matrix
        embed_weight = self.get_esm_embedding_weight()  # [vocab_size, 2560]

        # 2. Build one-hot vectors for special tokens
        cls_idx = esm_alphabet.cls_idx
        eos_idx = esm_alphabet.eos_idx

        cls_onehot = torch.zeros(1, vocab_size, device=device).to(soft_sequence.dtype)
        cls_onehot[0, cls_idx] = 1.0

        eos_onehot = torch.zeros(1, vocab_size, device=device).to(soft_sequence.dtype)
        eos_onehot[0, eos_idx] = 1.0

        # 3. Concatenate and convert to embeddings
        soft_sequence_with_special = torch.cat([
            cls_onehot,     # [1, vocab_size]
            soft_sequence,  # [L, vocab_size]
            eos_onehot      # [1, vocab_size]
        ], dim=0)  # [L+2, vocab_size]

        soft_sequence_with_special = soft_sequence_with_special.to(embed_weight.dtype)
        # ESM-2 scales embeddings by sqrt(d_model); omitting this would make inputs ~50× smaller
        # than during pre-training.
        soft_embeddings = (
            torch.matmul(soft_sequence_with_special, embed_weight) * self.model.esm.embed_scale
        )
        soft_embeddings = soft_embeddings.unsqueeze(0)  # [1, L+2, 2560]

        # 4. Manual ESM-2 forward (collect all layer representations)
        esm_output = self._esm_forward_from_embeddings(self.model.esm, soft_embeddings)

        # 5. Stack layer representations for weighted fusion
        esm_s = torch.stack(
            [v for _, v in sorted(esm_output["representations"].items())], dim=2
        )  # [1, L+2, n_layers+1, 2560]

        # Remove <cls> and <eos> tokens
        esm_s = esm_s[:, 1:-1]  # [1, L, n_layers+1, 2560]

        esm_s = esm_s.to(dtype)

        # 6. Compute pairwise representation (esm_z) and project via esm_z_mlp
        if self.model.cfg.use_esm_attn_map:
            esm_z = esm_output["attentions"]
            # Official permutation: [B, n_layers, n_heads, T, T]
            #   → [B, T, T, n_layers, n_heads] → [B, T, T, n_layers*n_heads]
            esm_z = esm_z.permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            esm_z = esm_z.to(dtype)
            s_z_0 = self.model.esm_z_mlp(esm_z)
        else:
            # Mimic esmfold.py L208: use a zero matrix when attention maps are disabled
            B, L = 1, soft_sequence.shape[0]
            s_z_0 = esm_s.new_zeros(B, L, L, self.model.cfg.trunk.pairwise_state_dim)

        if os.environ.get("DEBUG_ESMFOLD") == "1":
            print(f"[DEBUG] esm_s norm: {esm_s.norm().item():.4f}, mean: {esm_s.mean().item():.4f}")

        # Apply softmax-weighted combination across layers
        # [1, L, n_layers+1, 2560] * [1, 1, n_layers+1, 1] → [1, L, 2560]
        w = torch.softmax(self.model.esm_s_combine, dim=0).view(1, 1, -1, 1)
        esm_s = torch.sum(esm_s * w, dim=2)

        # 7. MLP projection and amino-acid embedding
        s_s_0 = self.model.esm_s_mlp(esm_s)  # [1, L, 1024]

        if os.environ.get("DEBUG_ESMFOLD") == "1":
            print(f"[DEBUG] s_s_0 (after mlp) norm: {s_s_0.norm().item():.4f}")

        '''
        Strategy for differentiable amino-acid embedding via STE:
        - Forward:  use argmax to obtain a hard sequence (trunk requires integer aatype).
        - Backward: gradient flows through the soft sequence via STE + matrix multiply.
        '''
        # Step 1: Hard sequence indices in ESM vocabulary
        hard_indices = torch.argmax(soft_sequence, dim=-1)  # [L]

        # Step 2: Build ESM-vocab → AF2-vocab index mapping.
        # af2_to_esm is shifted by 1 (index 0 = padding); subtract 1 for the true AF2 index.
        esm_to_af2_idx = torch.full((vocab_size,), 20, dtype=torch.long, device=device)  # 20 = unknown (X)
        for af2_idx_shifted, esm_idx in enumerate(self.model.af2_to_esm):
            if af2_idx_shifted > 0:  # skip padding (index 0)
                esm_to_af2_idx[esm_idx] = af2_idx_shifted - 1  # true AF2 index in [0, 20]

        '''
        esm_to_af2_idx:
        tensor([20, 20, 20, 20, 10,  0,  7, 19, 15,  6,  1, 16,  9,  3, 14, 11,  5,  2,
        13, 18, 12,  8, 17,  4, 20, 20, 20, 20, 20, 20, 20, 20, 20])
        '''

        # Step 3: Convert to AF2 indices for the trunk
        aa_af2 = esm_to_af2_idx[hard_indices].unsqueeze(0)  # [1, L]

        # Step 4: Hard one-hot in ESM vocabulary
        hard_onehot = torch.zeros_like(soft_sequence)  # [L, vocab_size]
        hard_onehot.scatter_(1, hard_indices.unsqueeze(1), 1.0)

        # Step 5: STE — forward uses hard, backward uses soft
        ste_sequence = (hard_onehot - soft_sequence).detach() + soft_sequence
        # Forward:  ste_sequence == hard_onehot (first term is constant due to detach)
        # Backward: ∂ste_sequence/∂soft_sequence == 1 (straight-through)

        # Step 6: Build ESM-vocab → AF2-vocab mapping matrix for matrix multiply
        # mapping_matrix[i, j] = 1 if ESM index i maps to AF2 index j
        n_af2_tokens = self.model.embedding.num_embeddings
        mapping_matrix = torch.zeros(vocab_size, n_af2_tokens,
                                     device=device, dtype=ste_sequence.dtype)
        for esm_idx in range(vocab_size):
            af2_idx = esm_to_af2_idx[esm_idx].item()
            if af2_idx < n_af2_tokens:
                mapping_matrix[esm_idx, af2_idx] = 1.0

        # Step 7: Compute soft AA embedding via fully differentiable matrix multiply
        # ste_sequence:    [L, vocab_esm]
        # mapping_matrix:  [vocab_esm, vocab_af2]
        # embedding.weight:[vocab_af2, c_s]
        embed_weight = self.model.embedding.weight  # [vocab_af2, c_s]

        # Cast to model precision to avoid RuntimeError (casting is differentiable in PyTorch)
        ste_sequence_casted = ste_sequence.to(dtype)
        mapping_matrix_casted = mapping_matrix.to(dtype)

        # [L, vocab_esm] @ [vocab_esm, vocab_af2] @ [vocab_af2, c_s] = [L, c_s]
        ste_sequence_af2 = torch.matmul(ste_sequence_casted, mapping_matrix_casted)  # [L, vocab_af2]
        soft_aa_embed = torch.matmul(ste_sequence_af2, embed_weight)  # [L, c_s]
        soft_aa_embed = soft_aa_embed.unsqueeze(0)  # [1, L, c_s]

        if os.environ.get("DEBUG_ESMFOLD") == "1":
            print(f"[DEBUG] soft_aa_embed norm: {soft_aa_embed.norm().item():.4f}")

        # Step 8: Add soft AA embedding (gradients now flow back to soft_sequence)
        s_s_0 = s_s_0 + soft_aa_embed

        # 8. Prepare trunk inputs
        B, L = aa_af2.shape
        residx = torch.arange(L, device=device).expand_as(aa_af2)
        mask = torch.ones_like(aa_af2).to(dtype)

        if os.environ.get("DEBUG_ESMFOLD") == "1":
            print(f"[DEBUG] s_s_0 shape: {s_s_0.shape}")
            print(f"[DEBUG] s_z_0 shape: {s_z_0.shape}")
            print(f"[DEBUG] aa_af2 shape: {aa_af2.shape}, range: {aa_af2.min().item()}-{aa_af2.max().item()}")
            print(f"[DEBUG] residx shape: {residx.shape}, range: {residx.min().item()}-{residx.max().item()}")
            print(f"[DEBUG] mask shape: {mask.shape}")

        # 9. Forward through folding trunk
        # no_recycles=0 → 1 pass (lowest memory); no_recycles=3 → 4 passes (highest quality)
        torch.cuda.empty_cache()  # flush allocator cache to avoid fragmentation OOM
        output = self.model.trunk(
            s_s_0,
            s_z_0,
            aa_af2.long(),
            residx.long(),
            mask,
            no_recycles=self.no_recycles
        )

        # 10. Extract pLDDT
        # output["states"]: [num_recycles, B, L, 384]
        lddt_logits = self.model.lddt_head(output["states"])
        lddt_logits = lddt_logits.reshape(
            output["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        # lddt_logits: [recycles, B, L, 37, 50]

        # Use the last recycle pass; result in 0–1 range, then scale to 0–100
        plddt = categorical_lddt(lddt_logits[-1], bins=self.model.lddt_bins)
        plddt = 100 * plddt  # [B, L, 37]

        # 11. Compute mean_plddt, aligned with official esmfold.py line 334
        structure = {
            "positions": output["positions"][-1],  # [B, L, 14, 3]
            "aatype": aa_af2,                       # [B, L]
        }
        # make_atom14_masks adds atom37_atom_exists [B, L, 37] to structure
        make_atom14_masks(structure)

        # Combine atom existence mask with sequence mask
        atom37_mask = structure["atom37_atom_exists"] * mask.unsqueeze(-1)

        # Official formula: mean_plddt = (plddt * atom37_mask).sum() / atom37_mask.sum()
        sum_plddt = (plddt * atom37_mask).sum(dim=(1, 2))
        sum_mask = atom37_mask.sum(dim=(1, 2))
        mean_plddt = (sum_plddt / (sum_mask + 1e-8))[0]  # [1] → scalar

        # 12. Extract pTM
        # output["s_z"]: [B, L, L, 128]
        ptm_logits = self.model.ptm_head(output["s_z"])
        seqlen = mask.type(torch.int64).sum(1)
        ptm_values = torch.stack([
            compute_tm(
                logits[None, :sl, :sl],
                max_bins=31,
                no_bins=self.model.distogram_bins,
            )
            for logits, sl in zip(ptm_logits, seqlen)
        ])
        ptm_value = ptm_values[0]

        return {
            'mean_plddt': mean_plddt,
            'ptm': ptm_value,
            'plddt': plddt[0],               # [L, 37]
            'positions': output["positions"][-1]
        }

    def predict_structure(self,
                          sequence: str,
                          return_pdb: bool = False) -> Dict:
        """
        Predict the structure of a single sequence.

        Args:
            sequence:   Protein sequence (amino acid string).
            return_pdb: Whether to return the structure in PDB format.

        Returns:
            result: Dict with the following fields:
                - plddt:      [L] per-residue pLDDT scores.
                - mean_plddt: Mean pLDDT score.
                - ptm:        pTM (predicted TM-score).
                - pdb:        PDB-format structure string (only if return_pdb=True).
        """
        with torch.no_grad():
            output = self.model.infer([sequence])

            # output['plddt']: [1, L, 37] — predicted values for all atom types
            plddt_atom37 = output['plddt'][0]
            if plddt_atom37.dtype == torch.bfloat16:
                plddt_atom37 = plddt_atom37.float()

            # Global mean pLDDT (pre-computed by ESMFold with atom mask applied)
            mean_plddt = float(output['mean_plddt'][0].cpu().item())

            # Per-residue pLDDT: average over valid atoms for each residue
            atom37_mask = output["atom37_atom_exists"][0]
            plddt_res = (plddt_atom37 * atom37_mask).sum(dim=-1) / (atom37_mask.sum(dim=-1) + 1e-8)
            plddt = plddt_res.cpu().numpy()  # [L]

            # pTM score — output['ptm']: [1]
            ptm_tensor = output['ptm'][0]
            if ptm_tensor.dtype == torch.bfloat16:
                ptm_tensor = ptm_tensor.float()
            ptm = float(ptm_tensor.cpu().item())

            result = {
                'plddt': plddt,
                'mean_plddt': mean_plddt,
                'ptm': ptm,
            }

            if return_pdb:
                pdb_string = self.model.output_to_pdb(output)[0]
                result['pdb'] = pdb_string

            return result


class LightweightStructureEvaluator:
    """
    Lightweight structure evaluator.

    Uses ESM-2 perplexity as a coarse proxy for structural quality.
    Much faster than ESMFold; suitable for early optimization stages.
    """

    def __init__(self, esm_manager, device: str = 'cuda'):
        """
        Initialize the lightweight evaluator.

        Args:
            esm_manager: ESM2Manager instance.
            device:      Compute device.
        """
        self.esm_manager = esm_manager
        self.device = device

    def compute_perplexity(self, sequence: str) -> float:
        """
        Compute the perplexity of a sequence (lower is better).

        Args:
            sequence: Protein sequence.

        Returns:
            perplexity: Perplexity value.
        """
        with torch.no_grad():
            tokens = self.esm_manager.alphabet.encode(sequence)
            tokens_tensor = torch.tensor([tokens], device=self.device)

            results = self.esm_manager.model(tokens_tensor, repr_layers=[36])

            # Simplified placeholder — a proper implementation requires the ESM-2 MLM head.
            # TODO: implement MLM-based perplexity
            perplexity = 1.0

            return perplexity

    def get_pseudo_rewards(self,
                           sequences: List[str],
                           baseline_perplexity: float = 10.0) -> torch.Tensor:
        """
        Compute pseudo-rewards based on perplexity (lower perplexity → higher reward).

        Args:
            sequences:            List of protein sequences.
            baseline_perplexity:  Baseline perplexity value.

        Returns:
            rewards: [G] pseudo-reward tensor.
        """
        perplexities = [self.compute_perplexity(seq) for seq in sequences]
        perplexities = torch.tensor(perplexities, dtype=torch.float32)
        rewards = baseline_perplexity - perplexities
        return rewards


class AdaptiveStructureEvaluator:
    """
    Adaptive structure evaluator.

    Uses different evaluation strategies at different stages of optimization:
    - Early stage:  lightweight evaluation (fast).
    - Middle stage: ESMFold every N steps.
    - Late stage:   ESMFold every step (accurate).
    """

    def __init__(self,
                 esmfold_evaluator: Optional[ESMFoldEvaluator] = None,
                 lightweight_evaluator: Optional[LightweightStructureEvaluator] = None,
                 early_stage_iters: int = 100,
                 late_stage_iters: int = 50,
                 esmfold_interval: int = 10):
        """
        Initialize the adaptive evaluator.

        Args:
            esmfold_evaluator:      ESMFold evaluator instance.
            lightweight_evaluator:  Lightweight evaluator instance.
            early_stage_iters:      Number of iterations in the early stage.
            late_stage_iters:       Number of iterations in the late stage.
            esmfold_interval:       ESMFold evaluation interval during the middle stage.
        """
        self.esmfold_evaluator = esmfold_evaluator
        self.lightweight_evaluator = lightweight_evaluator
        self.early_stage_iters = early_stage_iters
        self.late_stage_iters = late_stage_iters
        self.esmfold_interval = esmfold_interval

    def should_use_esmfold(self, iteration: int, total_iterations: int) -> bool:
        """
        Determine whether ESMFold should be used at the current iteration.

        Args:
            iteration:        Current iteration index.
            total_iterations: Total number of iterations.

        Returns:
            True if ESMFold should be used, False otherwise.
        """
        # Early stage: skip ESMFold
        if iteration < self.early_stage_iters:
            return False

        # Late stage: use ESMFold every step
        if iteration >= total_iterations - self.late_stage_iters:
            return True

        # Middle stage: use ESMFold every N steps
        return iteration % self.esmfold_interval == 0

    def evaluate(self,
                 sequences: List[str],
                 iteration: int,
                 total_iterations: int,
                 reward_type: str = 'plddt') -> Tuple[torch.Tensor, str]:
        """
        Adaptive evaluation of a list of sequences.

        Args:
            sequences:        List of protein sequences.
            iteration:        Current iteration index.
            total_iterations: Total number of iterations.
            reward_type:      Reward metric type.

        Returns:
            rewards:        [G] reward tensor.
            evaluator_type: Which evaluator was used ('esmfold', 'lightweight', or 'none').
        """
        use_esmfold = self.should_use_esmfold(iteration, total_iterations)

        if use_esmfold and self.esmfold_evaluator is not None:
            rewards, _ = self.esmfold_evaluator.get_rewards(
                sequences, reward_type=reward_type, show_progress=False
            )
            evaluator_type = 'esmfold'
        elif self.lightweight_evaluator is not None:
            rewards = self.lightweight_evaluator.get_pseudo_rewards(sequences)
            evaluator_type = 'lightweight'
        else:
            rewards = torch.zeros(len(sequences), dtype=torch.float32)
            evaluator_type = 'none'
            warnings.warn("No structure evaluator available; returning zero rewards.")

        return rewards, evaluator_type

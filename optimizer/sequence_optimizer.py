import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from .esmfold_evaluator import ESMFoldEvaluator

class ProteinSequenceLogits(nn.Module):
    """
    Trainable protein sequence logits representation.

    Core design: only optimize over the 20 standard amino acids.
    - Trainable parameter: P_canonical [L, 20]
    - At each forward pass, a P_full [L, vocab_size] base matrix (filled with -1e9)
      is created and P_canonical is scattered into the 20 canonical amino acid positions.
    - Gumbel-Softmax is applied to P_full, ensuring illegal tokens have zero probability.
    - Gradients flow only through the 20 canonical positions, updating only P_canonical.
    """

    def __init__(self, seq_length, vocab_size=33, init_sequence=None, esm_alphabet=None,
                 fixed_positions=None, decoupled_ste=False):
        """
        Args:
            seq_length: Sequence length L.
            vocab_size: ESM-2 vocabulary size (33 amino acids + special tokens).
            init_sequence: Optional initial sequence (string or token index list) for warm start.
            esm_alphabet: ESM-2 alphabet object used for sequence encoding.
            fixed_positions: Dict specifying positions to fix and their amino acids,
                             e.g. {0: 'M', 5: 'A', 10: 'G'}.
            decoupled_ste: Whether to decouple the sampling temperature from the gradient temperature.
        """
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.esm_alphabet = esm_alphabet
        self.fixed_positions = fixed_positions if fixed_positions is not None else {}
        self.decoupled_ste = decoupled_ste

        # 20 standard amino acids (AF2 vocabulary order)
        self.standard_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.num_canonical = len(self.standard_aa)  # 20

        # Map standard amino acids to their indices in the ESM-2 alphabet
        if esm_alphabet is not None:
            self.canonical_indices = torch.tensor([
                esm_alphabet.get_idx(aa) for aa in self.standard_aa
            ], dtype=torch.long)
            self.vocab_size = len(esm_alphabet)
        else:
            # Default indices (ESM-2 standard layout, remapped to AF2 order):
            # A:5, R:10, N:17, D:13, C:23, Q:16, E:9, G:6, H:21, I:12,
            # L:4,  K:15, M:20, F:18, P:14, S:8, T:11, W:22, Y:19, V:7
            self.canonical_indices = torch.tensor(
                [5, 10, 17, 13, 23, 16, 9, 6, 21, 12,
                 4, 15, 20, 18, 14, 8, 11, 22, 19, 7],
                dtype=torch.long
            )
            self.vocab_size = vocab_size

        # Separate mutable and fixed position indices
        self._setup_mutable_and_fixed_indices()

        # Initialize logits: either from a given sequence or randomly
        if init_sequence is not None:
            logits_full_init = self._sequence_to_canonical_logits(init_sequence)
        else:
            logits_full_init = torch.randn(seq_length, self.num_canonical) * 0.1

        # Trainable parameters for mutable positions: P_trainable [L_mutable, 20]
        if len(self.mutable_indices) > 0:
            self.logits_trainable = nn.Parameter(
                logits_full_init[self.mutable_indices, :]
            )
        else:
            self.logits_trainable = nn.Parameter(torch.empty(0, self.num_canonical))

        # Non-trainable buffer for fixed positions: P_fixed [L_fixed, 20]
        if len(self.fixed_indices) > 0:
            self.register_buffer(
                'logits_fixed',
                logits_full_init[self.fixed_indices, :]
            )
        else:
            self.register_buffer('logits_fixed', torch.empty(0, self.num_canonical))

    def _setup_mutable_and_fixed_indices(self):
        """Set up sorted index lists for mutable and fixed positions."""
        all_positions = set(range(self.seq_length))
        fixed_pos_set = set(self.fixed_positions.keys())
        mutable_pos_set = all_positions - fixed_pos_set

        self.mutable_indices = sorted(list(mutable_pos_set))
        self.fixed_indices = sorted(list(fixed_pos_set))

        # Build a mapping from fixed position → canonical index
        if len(self.fixed_indices) > 0:
            self.fixed_aa_canonical_indices = {}
            for pos, aa in self.fixed_positions.items():
                if aa in self.standard_aa:
                    self.fixed_aa_canonical_indices[pos] = self.standard_aa.index(aa)
                else:
                    raise ValueError(f"Fixed position {pos} has non-standard amino acid '{aa}'")

    def _sequence_to_canonical_logits(self, sequence):
        """
        Convert a discrete sequence to canonical amino acid logits [L, 20].

        Args:
            sequence: String (e.g. "ACDEFG...") or list of full-vocabulary token indices.

        Returns:
            logits_canonical: [L, 20] logits over the 20 standard amino acids.
        """
        logits_canonical = torch.zeros(self.seq_length, self.num_canonical)

        if isinstance(sequence, str):
            if self.esm_alphabet is None:
                raise ValueError("esm_alphabet must be provided to convert a string sequence to token indices")
            sequence = [self.esm_alphabet.get_idx(aa) for aa in sequence]

        # Build a mapping from full-vocabulary index → canonical index
        full_to_canonical = {
            full_idx: canonical_idx
            for canonical_idx, full_idx in enumerate(self.canonical_indices.tolist())
        }

        for i, token_idx_full in enumerate(sequence):
            if i >= self.seq_length:
                break
            if token_idx_full in full_to_canonical:
                canonical_idx = full_to_canonical[token_idx_full]
                logits_canonical[i, canonical_idx] = 8.0
            else:
                raise ValueError(f"Sequence contains non-standard amino acid token: {token_idx_full}")
            # Add a small amount of noise for better initialization
            logits_canonical[i, :] += torch.randn(self.num_canonical) * 0.01

        return logits_canonical

    def _reconstruct_full_logits_canonical(self):
        """
        Dynamically reconstruct the full logits_canonical tensor [L, 20].

        Steps:
        1. Create an empty [L, 20] template.
        2. Scatter logits_trainable [L_mutable, 20] into mutable positions (with gradient).
        3. Scatter logits_fixed [L_fixed, 20] into fixed positions using extreme values
           (1e9 for the fixed amino acid, -1e9 for others) to enforce hard selection.
        Gradients flow automatically only through logits_trainable.

        Returns:
            logits_canonical: [L, 20] full canonical amino acid logits.
        """
        device = self.logits_trainable.device

        logits_canonical = torch.zeros(
            self.seq_length, self.num_canonical,
            dtype=self.logits_trainable.dtype,
            device=device
        )

        if len(self.mutable_indices) > 0:
            logits_canonical[self.mutable_indices, :] = self.logits_trainable

        if len(self.fixed_indices) > 0:
            for pos in self.fixed_indices:
                canonical_idx = self.fixed_aa_canonical_indices[pos]
                fixed_logit = torch.full((self.num_canonical,), -1e9, device=device)
                fixed_logit[canonical_idx] = 1e9
                logits_canonical[pos, :] = fixed_logit

        return logits_canonical

    def _sample_gumbel(self, shape, device, eps=1e-20):
        """
        Sample from Gumbel(0, 1) using the current PyTorch random state.

        Args:
            shape: Output shape.
            device: Target device.
            eps: Numerical stability constant.

        Returns:
            Gumbel noise tensor of the given shape.
        """
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, temperature=1.0, hard=False):
        """
        Sample a soft sequence using Gumbel-Softmax.

        Steps:
        1. Reconstruct full logits_canonical [L, 20].
        2. Create a [L, vocab_size] base matrix filled with -1e9.
        3. Scatter the canonical logits into their corresponding vocabulary positions.
        4. Apply manual Gumbel-Softmax; illegal tokens have near-zero probability.
        5. Gradients flow only through the 20 canonical positions.

        Returns:
            soft_sequence: [L, vocab_size] soft one-hot representation.
        """
        logits_canonical = self._reconstruct_full_logits_canonical()
        device = logits_canonical.device

        logits_full = torch.full(
            (self.seq_length, self.vocab_size),
            -1e9,
            dtype=logits_canonical.dtype,
            device=device
        )

        canonical_indices = self.canonical_indices.to(device)
        indices_expanded = canonical_indices.unsqueeze(0).expand(self.seq_length, -1)  # [L, 20]
        logits_full.scatter_(dim=1, index=indices_expanded, src=logits_canonical)

        gumbel_noise = self._sample_gumbel(logits_full.shape, device)
        y = logits_full + gumbel_noise

        if self.decoupled_ste and temperature < 0.8:
            # Forward sampling uses the actual temperature; backward gradients use tau=0.8
            # to prevent gradient explosion at low temperatures.
            y_sample = F.softmax(y / temperature, dim=-1)
            y_grad_proxy = F.softmax(y / 0.8, dim=-1)

            if hard:
                y_hard = torch.zeros_like(y_sample)
                y_hard.scatter_(1, y_sample.argmax(dim=-1, keepdim=True), 1.0)
                # STE: forward y_hard, backward through y_grad_proxy
                soft_sequence = (y_hard - y_grad_proxy).detach() + y_grad_proxy
            else:
                soft_sequence = (y_sample - y_grad_proxy).detach() + y_grad_proxy
        else:
            soft_sequence = F.softmax(y / temperature, dim=-1)

            if hard:
                # Straight-through estimator
                y_hard = torch.zeros_like(soft_sequence)
                y_hard.scatter_(1, soft_sequence.argmax(dim=-1, keepdim=True), 1.0)
                soft_sequence = (y_hard - soft_sequence).detach() + soft_sequence

        return soft_sequence  # [L, vocab_size]

    def get_discrete_sequence(self):
        """
        Obtain the discrete sequence via argmax.

        Steps:
        0. Reconstruct full logits_canonical [L, 20].
        1. Create a [L, vocab_size] base matrix filled with -inf.
        2. Scatter canonical logits into their vocabulary positions.
        3. argmax — only standard amino acids can be selected; fixed positions
           are guaranteed to select their assigned amino acid due to extreme logit values.

        Returns:
            discrete_sequence: [L] discrete token indices (full vocabulary).
        """
        logits_canonical = self._reconstruct_full_logits_canonical()
        device = logits_canonical.device

        logits_full = torch.full(
            (self.seq_length, self.vocab_size),
            float('-inf'),
            dtype=logits_canonical.dtype,
            device=device
        )

        canonical_indices = self.canonical_indices.to(device)
        indices_expanded = canonical_indices.unsqueeze(0).expand(self.seq_length, -1)
        logits_full.scatter_(dim=1, index=indices_expanded, src=logits_canonical)

        discrete_sequence = torch.argmax(logits_full, dim=-1)  # [L]
        return discrete_sequence

    def get_full_logits_matrix_with_probs(self):
        """
        Return the full logits matrix and corresponding probability distribution
        for all positions (both mutable and fixed).

        Returns:
            logits_canonical: [L, 20] full canonical amino acid logits.
            probs_canonical:  [L, 20] probability distribution (softmax over mutable,
                              one-hot for fixed positions).
            aa_mapping:       list[str] of the 20 standard amino acids in canonical order.
        """
        logits_canonical = self._reconstruct_full_logits_canonical()
        probs_canonical = torch.zeros_like(logits_canonical)

        if len(self.mutable_indices) > 0:
            mutable_logits = logits_canonical[self.mutable_indices, :]
            probs_canonical[self.mutable_indices, :] = F.softmax(mutable_logits, dim=-1)

        if len(self.fixed_indices) > 0:
            for pos in self.fixed_indices:
                canonical_idx = self.fixed_aa_canonical_indices[pos]
                probs_canonical[pos, :] = 0.0
                probs_canonical[pos, canonical_idx] = 1.0

        return logits_canonical, probs_canonical, self.standard_aa


class SequenceOptimizer:
    """Main sequence optimizer supporting multi-model ensembles, penalty terms, and ESMFold structural constraints."""

    def __init__(self,
                 model,                # Trained OmniESI_WithESM model or list of models
                 esm_manager,          # ESM2Manager
                 seq_length,           # Sequence length
                 molecule_graph,       # Fixed small-molecule graph
                 init_sequence=None,
                 fixed_positions=None, # Fixed position dict {pos: 'AA'}
                 device=None,
                 penalty_type=None,    # Penalty type: 'kl', 'l2_embedding', or None
                 penalty_lambda=0.0,   # Penalty coefficient
                 # ESMFold structural constraint parameters
                 use_esmfold=False,
                 esmfold_evaluator=None,
                 lambda_struct=0.5,
                 plddt_threshold=70.0,
                 ptm_threshold=0.5,
                 structure_loss_type='plddt',    # 'plddt', 'ptm', or 'combined'
                 structure_loss_function='relu', # 'relu', 'softplus', or 'soft_hinge'
                 softplus_sigma=1.0,
                 soft_hinge_margin=5.0,
                 use_ste=False,
                 decoupled_ste=False,
                 kcat_loss_type='mean_only',      # 'mean_only' or 'uncertainty_aware'
                 kcat_uncertainty_beta=0.1):

        if isinstance(model, list):
            self.models = model
            self.is_ensemble = True
        else:
            self.models = [model]
            self.is_ensemble = False

        self.model = self.models[0]
        self.esm_manager = esm_manager
        self.seq_length = seq_length
        self.init_sequence = init_sequence

        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda

        self.use_esmfold = use_esmfold
        self.esmfold_evaluator = esmfold_evaluator
        self.lambda_struct = lambda_struct
        self.plddt_threshold = plddt_threshold
        self.ptm_threshold = ptm_threshold
        self.structure_loss_type = structure_loss_type
        self.structure_loss_function = structure_loss_function
        self.softplus_sigma = softplus_sigma
        self.soft_hinge_margin = soft_hinge_margin
        self.use_ste = use_ste
        self.decoupled_ste = decoupled_ste
        self.kcat_loss_type = kcat_loss_type
        self.kcat_uncertainty_beta = kcat_uncertainty_beta

        # Validate ESMFold configuration
        if self.use_esmfold:
            from .esmfold_evaluator import ESMFOLD_AVAILABLE

            if not ESMFOLD_AVAILABLE:
                print("[Warning] ESMFold dependencies not found (esm, openfold, etc.). "
                      "Structural constraint disabled. See esmfold_evaluator.py for installation instructions.")
                self.use_esmfold = False
            elif self.esmfold_evaluator is None:
                print("[Warning] use_esmfold=True but no esmfold_evaluator instance provided. "
                      "Structural constraint disabled.")
                self.use_esmfold = False
            elif not isinstance(self.esmfold_evaluator, ESMFoldEvaluator):
                actual_type = type(self.esmfold_evaluator).__name__
                print(f"[Error] esmfold_evaluator has unexpected type "
                      f"(expected: ESMFoldEvaluator, got: {actual_type}). "
                      f"Structural constraint disabled.")
                self.use_esmfold = False

        if device is None:
            self.device = next(self.models[0].parameters()).device
        else:
            self.device = device

        self.molecule_graph = molecule_graph.to(self.device)

        # Freeze all model parameters
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False

        self.seq_logits = ProteinSequenceLogits(
            seq_length=seq_length,
            vocab_size=len(esm_manager.alphabet),
            init_sequence=init_sequence,
            esm_alphabet=esm_manager.alphabet,
            fixed_positions=fixed_positions,
            decoupled_ste=decoupled_ste
        ).to(self.device)

        # ESM-2 token embedding matrix (moved to the target device once)
        self.token_embedding_weight = esm_manager.get_token_embedding().float().to(self.device)

        # Pre-compute one-hot representation of the original sequence (for KL penalty)
        if init_sequence is not None:
            self._init_original_onehot()
            if penalty_type == 'l2_embedding':
                self._init_original_embedding()

        # Cache: stores kcat / plddt / ptm for previously evaluated sequences
        # key: sequence string, value: {'kcat': float, 'plddt': float, 'ptm': float}
        self._sequence_cache = {}

    def _init_original_onehot(self):
        """
        Initialize the one-hot representation of the original sequence
        over the 20 standard amino acids, used for the KL divergence penalty.
        """
        self.original_onehot = torch.zeros(
            self.seq_length, self.seq_logits.num_canonical,
            device=self.device
        )
        for i, aa in enumerate(self.init_sequence):
            if aa in self.seq_logits.standard_aa:
                canonical_idx = self.seq_logits.standard_aa.index(aa)
                self.original_onehot[i, canonical_idx] = 1.0

    def _init_original_embedding(self):
        """
        Pre-compute the ESM-2 embedding of the original sequence,
        used for the L2 norm embedding penalty.
        """
        with torch.no_grad():
            original_onehot_full = torch.zeros(
                self.seq_length, len(self.esm_manager.alphabet),
                device=self.device
            )
            for i, aa in enumerate(self.init_sequence):
                idx = self.esm_manager.alphabet.get_idx(aa)
                original_onehot_full[i, idx] = 1.0

            self.original_embedding = self._get_esm2_embedding(original_onehot_full)

    def _get_esm2_embedding(self, soft_sequence):
        """
        Compute the ESM-2 embedding for a soft sequence (without special-token embeddings
        included in the output).

        Args:
            soft_sequence: [L, vocab_size] soft one-hot representation.

        Returns:
            embedding: [L, 2560] ESM-2 output embeddings.
        """
        device = soft_sequence.device
        vocab_size = soft_sequence.shape[-1]

        cls_idx = self.esm_manager.alphabet.cls_idx
        eos_idx = self.esm_manager.alphabet.eos_idx

        bos_onehot = torch.zeros(1, vocab_size, device=device)
        bos_onehot[0, cls_idx] = 1.0

        eos_onehot = torch.zeros(1, vocab_size, device=device)
        eos_onehot[0, eos_idx] = 1.0

        # Concatenate: <cls> + sequence + <eos>
        soft_sequence_with_special = torch.cat([
            bos_onehot,     # (1, 33)
            soft_sequence,  # (L, 33)
            eos_onehot      # (1, 33)
        ], dim=0)  # (L+2, 33)

        soft_embeddings = torch.matmul(
            soft_sequence_with_special,
            self.token_embedding_weight
        )  # (L+2, 2560)

        soft_embeddings = soft_embeddings.unsqueeze(0)  # (1, L+2, 2560)

        esm_output_with_special = self.esm_manager.forward_from_embeddings(soft_embeddings)

        esm_output = esm_output_with_special[:, 1:-1, :]  # (1, L, 2560)
        return esm_output.squeeze(0)  # (L, 2560)

    def compute_kl_penalty(self, soft_sequence_canonical):
        """
        Compute the KL divergence penalty term, evaluated only at mutable positions.

        KL(P || Q) = sum(P * log(P / Q))
        where P is the optimized softmax distribution and Q is the original one-hot.
        Fixed positions are excluded because they are already constrained.

        Args:
            soft_sequence_canonical: [L, 20] logits over canonical amino acids.

        Returns:
            kl_penalty: Scalar KL divergence penalty.
        """
        if len(self.seq_logits.mutable_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        mutable_logits = soft_sequence_canonical[self.seq_logits.mutable_indices, :]
        mutable_target = self.original_onehot[self.seq_logits.mutable_indices, :]

        probs = F.softmax(mutable_logits, dim=-1)

        eps = 1e-8
        target_smooth = mutable_target + eps
        target_smooth = target_smooth / target_smooth.sum(dim=-1, keepdim=True)

        kl_div = probs * (torch.log(probs + eps) - torch.log(target_smooth))
        kl_penalty = kl_div.sum()

        return kl_penalty

    def compute_l2_embedding_penalty(self, soft_sequence):
        """
        Compute the ESM-2 embedding L2 norm penalty.

        Measures the L2 distance between the original sequence embedding
        and the current soft-sequence embedding.

        Args:
            soft_sequence: [L, vocab_size] optimized soft one-hot (full vocabulary).

        Returns:
            l2_penalty: Scalar L2 norm penalty.
        """
        current_embedding = self._get_esm2_embedding(soft_sequence)
        l2_penalty = torch.norm(current_embedding - self.original_embedding, p=2)
        return l2_penalty

    def get_hard_sequence_from_soft(self, soft_sequence):
        """
        Decode a hard sequence string from a soft sequence via argmax.

        Args:
            soft_sequence: [L, vocab_size] soft one-hot representation.

        Returns:
            sequence_str: Hard-decoded amino acid sequence string.
        """
        hard_indices = torch.argmax(soft_sequence, dim=-1)  # [L]
        alphabet = self.esm_manager.alphabet
        sequence_str = ''.join([
            alphabet.get_tok(idx.item())
            for idx in hard_indices.cpu()
        ])
        return sequence_str

    def compute_structure_loss(self, soft_sequence, use_ste=False):
        """
        Compute the ESMFold structural constraint loss with differentiable gradients.

        The soft_sequence has already been processed through STE (if applicable) in
        ProteinSequenceLogits.forward(), so it can be used directly here.

        Args:
            soft_sequence: [L, vocab_size] soft one-hot (with gradients, STE already applied).
            use_ste: Kept for interface compatibility; STE is handled upstream.

        Returns:
            struct_loss:  Scalar tensor, structural loss (differentiable).
            struct_score: float, structural score for logging (detached).
            plddt_value:  float, per-residue pLDDT mean (detached).
            ptm_value:    float, pTM score (detached).
        """
        if not self.use_esmfold or self.esmfold_evaluator is None:
            return torch.tensor(0.0, device=self.device), 0.0, None, None

        try:
            result = self.esmfold_evaluator.predict_structure_from_soft_embedding(
                soft_sequence=soft_sequence,
                esm_alphabet=self.esm_manager.alphabet
            )
            mean_plddt_tensor = result['mean_plddt']
            ptm_tensor = result['ptm']
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"[Critical Error] ESMFold soft-embedding evaluation failed")
            print(f"{'='*80}")
            print(f"Error: {e}")
            print(f"Soft sequence shape: {soft_sequence.shape}")
            print(f"Structure loss type: {self.structure_loss_type}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"ESMFold soft-embedding evaluation failed: {e}") from e

        if self.structure_loss_type == 'plddt':
            struct_score_tensor = mean_plddt_tensor
            threshold = self.plddt_threshold
        elif self.structure_loss_type == 'ptm':
            struct_score_tensor = ptm_tensor * 100
            threshold = self.ptm_threshold * 100
        elif self.structure_loss_type == 'combined':
            struct_score_tensor = 0.5 * mean_plddt_tensor + 0.5 * ptm_tensor * 100
            threshold = 0.5 * self.plddt_threshold + 0.5 * self.ptm_threshold * 100
        else:
            struct_score_tensor = mean_plddt_tensor
            threshold = self.plddt_threshold

        if self.structure_loss_function == 'softplus':
            # Softplus loss: σ * softplus((threshold - score) / σ)
            # Approaches linear when score < threshold; decays slowly (non-zero gradient) when score > threshold.
            sigma = self.softplus_sigma
            struct_loss_tensor = sigma * torch.nn.functional.softplus(
                (threshold - struct_score_tensor) / sigma
            )
        elif self.structure_loss_function == 'soft_hinge':
            # Soft-hinge (clamped softplus): smooth in [threshold, threshold + margin],
            # zero (with zero gradient) once score >= threshold + margin,
            # linear when score < threshold.
            # gap = threshold - score. Clamp from below at -margin so that for
            # score >= threshold + margin the clamp saturates (gradient = 0),
            # which kills the backward path through ESMFold for that region.
            sigma = self.softplus_sigma
            margin = self.soft_hinge_margin
            gap = threshold - struct_score_tensor
            gap_clamped = torch.clamp(gap, min=-margin)
            # Subtract softplus(-margin/sigma) so the loss is exactly 0 at the saturation point.
            offset = torch.nn.functional.softplus(
                torch.tensor(-margin / sigma, device=gap.device, dtype=gap.dtype)
            )
            struct_loss_tensor = sigma * (
                torch.nn.functional.softplus(gap_clamped / sigma) - offset
            )
        else:
            # ReLU / Hinge loss: max(0, threshold - score)
            struct_loss_tensor = torch.clamp(threshold - struct_score_tensor, min=0.0)

        struct_score = struct_score_tensor.detach().item()
        plddt_value = mean_plddt_tensor.detach().item()
        ptm_value = ptm_tensor.detach().item()

        return struct_loss_tensor, struct_score, plddt_value, ptm_value

    def soft_embedding_forward(self, soft_sequence):
        """
        Generate embeddings from a soft sequence and run a full model forward pass
        (supports multi-model ensemble).

        Steps:
        1. Append <cls> and <eos> special tokens.
        2. Soft embedding: soft_sequence @ ESM-2 token_embedding → (L+2, 2560).
        3. Forward through ESM-2 to extract protein representations.
        4. Prepare molecule graph and padding masks.
        5. Forward through OmniESI model.
        6. If ensemble, return the mean prediction across all models.

        Args:
            soft_sequence: (L, vocab_size) soft one-hot representation.

        Returns:
            score: Predicted kcat value. For ensemble mode returns the mean
                   (or a (mean, std) tuple when kcat_loss_type='uncertainty_aware').
        """
        device = soft_sequence.device
        vocab_size = soft_sequence.shape[-1]

        cls_idx = self.esm_manager.alphabet.cls_idx
        eos_idx = self.esm_manager.alphabet.eos_idx

        bos_onehot = torch.zeros(1, vocab_size, device=device)
        bos_onehot[0, cls_idx] = 1.0

        eos_onehot = torch.zeros(1, vocab_size, device=device)
        eos_onehot[0, eos_idx] = 1.0

        soft_sequence_with_special = torch.cat([
            bos_onehot,     # (1, 33)
            soft_sequence,  # (L, 33)
            eos_onehot      # (1, 33)
        ], dim=0)  # (L+2, 33)

        soft_embeddings = torch.matmul(
            soft_sequence_with_special,
            self.token_embedding_weight
        )  # (L+2, 2560)

        soft_embeddings = soft_embeddings.unsqueeze(0)  # (1, L+2, 2560)

        esm_output_with_special = self.esm_manager.forward_from_embeddings(soft_embeddings)
        esm_output = esm_output_with_special[:, 1:-1, :]  # (1, L, 2560)

        v_p = esm_output
        v_d = self.molecule_graph

        v_p_mask = torch.zeros(1, v_p.shape[1], dtype=torch.bool, device=self.device)
        v_d_mask = torch.zeros(1, v_d.num_nodes(), dtype=torch.bool, device=self.device)

        # drug_extractor internally consumes ndata['h'] via pop(), so we back up and
        # restore the node features around each model call to avoid KeyError on reuse.
        node_feats_backup = v_d.ndata['h']

        if self.is_ensemble:
            scores = []
            for model in self.models:
                v_d.ndata['h'] = node_feats_backup
                v_d_out, v_p_out, fusion_out, score = model(v_d, v_p, v_d_mask, v_p_mask)
                scores.append(score)

            v_d.ndata['h'] = node_feats_backup

            all_scores = torch.stack(scores, dim=0)  # [num_models, 1]

            if self.kcat_loss_type == 'uncertainty_aware':
                mean_score = all_scores.mean(dim=0)
                std_score = all_scores.std(dim=0)
                return mean_score, std_score
            else:
                return all_scores.mean(dim=0)
        else:
            v_d.ndata['h'] = node_feats_backup
            v_d_out, v_p_out, fusion_out, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
            v_d.ndata['h'] = node_feats_backup
            return score

    def optimize(self,
                 num_iterations=1000,
                 lr=0.001,
                 log_interval=50,
                 tau_init=1.0,
                 tau_min=0.5,
                 optimizer_type='adam',
                 optimizer_kwargs=None,
                 ste_interval=None,
                 ste_mode='none',
                 anneal=False,
                 annealing_scheme='linear',
                 print_opt=False,
                 early_stop=False,
                 patience=5,
                 output_dir=None,
                 seed=None,
                 lr_scheduler='none',
                 lr_min=1e-6):
        """
        Run the sequence optimization loop.

        Args:
            num_iterations:   Number of optimization iterations.
            lr:               Initial learning rate.
            log_interval:     Interval for console logging.
            tau_init:         Initial Gumbel-Softmax temperature.
            tau_min:          Minimum temperature (used when anneal=True).
            optimizer_type:   Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop', 'adagrad').
            optimizer_kwargs: Extra keyword arguments passed to the optimizer.
            ste_interval:     Apply STE every N iterations (None to disable).
            ste_mode:         STE mode ('hard' or 'none').
            anneal:           Whether to anneal the temperature over iterations.
            annealing_scheme: Annealing scheme ('linear', 'cosine', 'exp', 'fixed').
            print_opt:        Whether to write per-iteration CSV logs.
            early_stop:       Whether to enable early stopping.
            patience:         Number of consecutive unchanged sequences before stopping.
            output_dir:       Directory for CSV output.
            seed:             Random seed (used to suffix the CSV filename).
            lr_scheduler:     LR decay scheme ('none', 'linear', 'cosine').
            lr_min:           Minimum learning rate when lr_scheduler != 'none'.
        """
        optimizer = self._create_optimizer(
            optimizer_type=optimizer_type,
            lr=lr,
            optimizer_kwargs=optimizer_kwargs
        )

        scheduler = None
        if lr_scheduler != 'none':
            scheduler = self._create_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                num_iterations=num_iterations,
                lr_init=lr,
                lr_min=lr_min
            )

        history = []

        # CSV logging setup — always create CSV when output_dir is set;
        # print_opt only controls whether STE hard-encoding is used for pLDDT.
        csv_file = None
        if output_dir:
            csv_filename = f'kcat_iteration_{seed}.csv' if seed is not None else 'kcat_iteration.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            csv_file = open(csv_path, 'w')

            if self.use_esmfold:
                if self.structure_loss_type == 'plddt':
                    csv_file.write("iter,kcat_soft,kcat,kcat_std,plddt,grad_norm,kcat_grad,struct_grad,sequences,logit_sequences\n")
                elif self.structure_loss_type == 'ptm':
                    csv_file.write("iter,kcat_soft,kcat,kcat_std,ptm,grad_norm,kcat_grad,struct_grad,sequences,logit_sequences\n")
                elif self.structure_loss_type == 'combined':
                    csv_file.write("iter,kcat_soft,kcat,kcat_std,plddt,ptm,grad_norm,kcat_grad,struct_grad,sequences,logit_sequences\n")
                else:
                    csv_file.write("iter,kcat_soft,kcat,kcat_std,grad_norm,sequences,logit_sequences\n")
            else:
                csv_file.write("iter,kcat_soft,kcat,kcat_std,grad_norm,sequences,logit_sequences\n")

            print(f"\n[Logging] CSV logging enabled: {csv_path} (STE for pLDDT: {print_opt})")

        last_sequence_str = None
        patience_counter = 0

        print(f"\n[Optimization] Starting optimization")
        print(f"[Optimization] Iterations: {num_iterations}, lr: {lr}")
        if lr_scheduler != 'none':
            print(f"[Optimization] LR scheduler: {lr_scheduler} ({lr} -> {lr_min})")
        if anneal:
            print(f"[Optimization] Temperature annealing: {annealing_scheme} ({tau_init} -> {tau_min})")
        else:
            print(f"[Optimization] Fixed temperature: {tau_init}")
        print(f"[Optimization] Early stopping: {'enabled' if early_stop else 'disabled'} (patience={patience})")

        for iteration in range(num_iterations):
            # 1. Compute current temperature
            if anneal:
                if annealing_scheme == 'linear':
                    temperature = tau_init - (tau_init - tau_min) * (iteration / num_iterations)
                elif annealing_scheme == 'cosine':
                    temperature = tau_min + 0.5 * (tau_init - tau_min) * (
                        1 + np.cos(np.pi * iteration / num_iterations)
                    )
                elif annealing_scheme == 'exp':
                    temperature = tau_init * (tau_min / tau_init) ** (iteration / num_iterations)
                elif annealing_scheme == 'fixed':
                    temperature = tau_init
                else:
                    if iteration == 0:
                        print(f"[Warning] Unknown annealing scheme '{annealing_scheme}', using fixed temperature.")
                    temperature = tau_init
            else:
                temperature = tau_init

            temperature = max(temperature, 1e-5)

            # 2. Gumbel-Softmax sampling (STE handled inside seq_logits.forward)
            use_ste_this_iter = (
                ste_interval and ste_interval > 0 and (iteration + 1) % ste_interval == 0
            )
            use_hard = (use_ste_this_iter and ste_mode == 'hard') or self.use_ste
            soft_sequence = self.seq_logits(temperature=temperature, hard=use_hard)

            # Decode the hard sequence corresponding to this sampling step
            # (must be done before the logits are updated)
            current_iteration_indices = torch.argmax(soft_sequence, dim=-1)
            alphabet = self.esm_manager.alphabet
            current_iteration_seq_str = ''.join([
                alphabet.get_tok(idx.item()) for idx in current_iteration_indices.cpu()
            ])

            # 3. Forward pass to predict kcat
            if self.kcat_loss_type == 'uncertainty_aware' and self.is_ensemble:
                kcat_mean, kcat_std = self.soft_embedding_forward(soft_sequence)
                loss = -(kcat_mean - self.kcat_uncertainty_beta * kcat_std)
                kcat_pred = kcat_mean
            else:
                kcat_pred = self.soft_embedding_forward(soft_sequence)
                kcat_std = torch.tensor(0.0, device=self.device)
                loss = -kcat_pred

            # 4. Add penalty term
            penalty = torch.tensor(0.0, device=self.device)
            if self.penalty_type is not None and self.penalty_lambda > 0:
                if self.penalty_type == 'kl':
                    logits_canonical = self.seq_logits._reconstruct_full_logits_canonical()
                    penalty = self.compute_kl_penalty(logits_canonical)
                    loss = loss + self.penalty_lambda * penalty
                elif self.penalty_type == 'l2_embedding':
                    penalty = self.compute_l2_embedding_penalty(soft_sequence)
                    loss = loss + self.penalty_lambda * penalty

            # 5. Add ESMFold structural constraint loss
            struct_loss = torch.tensor(0.0, device=self.device)
            struct_score = 0.0
            plddt_value = None
            ptm_value = None
            if self.use_esmfold:
                # print_opt controls STE: True → hard-encode pLDDT; False → soft pLDDT
                struct_loss, struct_score, plddt_value, ptm_value = self.compute_structure_loss(
                    soft_sequence, use_ste=print_opt
                )
                loss = loss + self.lambda_struct * struct_loss

            # 6. Compute kcat for CSV logging.
            # When print_opt=True, compute hard-sequence kcat; otherwise use soft kcat directly.
            kcat_hard = kcat_pred.item()
            if print_opt and csv_file:
                if not self.use_ste:
                    if (current_iteration_seq_str in self._sequence_cache and
                            'kcat' in self._sequence_cache[current_iteration_seq_str]):
                        kcat_hard = self._sequence_cache[current_iteration_seq_str]['kcat']
                    else:
                        with torch.no_grad():
                            hard_onehot = F.one_hot(
                                current_iteration_indices,
                                num_classes=len(alphabet)
                            ).float().to(self.device)
                            res_hard = self.soft_embedding_forward(hard_onehot)
                            kcat_hard = res_hard[0].item() if isinstance(res_hard, tuple) else res_hard.item()
                        self._sequence_cache.setdefault(current_iteration_seq_str, {})['kcat'] = kcat_hard

            # 7. Backward pass and gradient computation
            kcat_grad_norm = 0.0
            struct_grad_norm = 0.0
            if self.use_esmfold and (iteration % log_interval == 0 or print_opt):
                # Isolated backward for kcat gradient norm
                optimizer.zero_grad()
                (-kcat_pred).backward(retain_graph=True)
                if self.seq_logits.logits_trainable.grad is not None:
                    kcat_grad_norm = self.seq_logits.logits_trainable.grad.norm().item()

                # Isolated backward for struct gradient norm
                optimizer.zero_grad()
                if struct_loss is not None:
                    (self.lambda_struct * struct_loss).backward(retain_graph=True)
                    if self.seq_logits.logits_trainable.grad is not None:
                        struct_grad_norm = self.seq_logits.logits_trainable.grad.norm().item()

            optimizer.zero_grad()
            loss.backward()

            total_grad_norm = 0.0
            if self.seq_logits.logits_trainable.grad is not None:
                total_grad_norm = self.seq_logits.logits_trainable.grad.norm().item()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Obtain the deterministic sequence (argmax of logits) after the update
            with torch.no_grad():
                updated_discrete_indices = self.seq_logits.get_discrete_sequence()
                updated_seq_str = ''.join([
                    alphabet.get_tok(idx.item()) for idx in updated_discrete_indices.cpu()
                ])

            # 8. Write CSV row (always when csv_file is open)
            if csv_file:
                if self.use_esmfold:
                    plddt_scalar = plddt_value
                    ptm_scalar = ptm_value
                    if self.structure_loss_type == 'plddt':
                        plddt_str = f"{plddt_scalar:.2f}" if plddt_scalar is not None else "N/A"
                        csv_file.write(f"{iteration + 1},{kcat_pred.item()},{kcat_hard},{kcat_std.item():.4f},{plddt_str},{total_grad_norm:.6f},{kcat_grad_norm:.6f},{struct_grad_norm:.6f},{current_iteration_seq_str},{updated_seq_str}\n")
                    elif self.structure_loss_type == 'ptm':
                        ptm_str = f"{ptm_scalar:.4f}" if ptm_scalar is not None else "N/A"
                        csv_file.write(f"{iteration + 1},{kcat_pred.item()},{kcat_hard},{kcat_std.item():.4f},{ptm_str},{total_grad_norm:.6f},{kcat_grad_norm:.6f},{struct_grad_norm:.6f},{current_iteration_seq_str},{updated_seq_str}\n")
                    elif self.structure_loss_type == 'combined':
                        plddt_str = f"{plddt_scalar:.2f}" if plddt_scalar is not None else "N/A"
                        ptm_str = f"{ptm_scalar:.4f}" if ptm_scalar is not None else "N/A"
                        csv_file.write(f"{iteration + 1},{kcat_pred.item()},{kcat_hard},{kcat_std.item():.4f},{plddt_str},{ptm_str},{total_grad_norm:.6f},{kcat_grad_norm:.6f},{struct_grad_norm:.6f},{current_iteration_seq_str},{updated_seq_str}\n")
                    else:
                        csv_file.write(f"{iteration + 1},{kcat_pred.item()},{kcat_hard},{kcat_std.item():.4f},{total_grad_norm:.6f},{current_iteration_seq_str},{updated_seq_str}\n")
                else:
                    csv_file.write(f"{iteration + 1},{kcat_pred.item()},{kcat_hard},{kcat_std.item():.4f},{total_grad_norm:.6f},{current_iteration_seq_str},{updated_seq_str}\n")
                csv_file.flush()

            # 9. Early stopping (based on argmax of logits, noise-free)
            discrete_seq_indices_clean = self.seq_logits.get_discrete_sequence()
            current_seq_str_clean = ''.join([
                alphabet.get_tok(idx.item()) for idx in discrete_seq_indices_clean.cpu()
            ])

            if early_stop:
                if last_sequence_str is not None and current_seq_str_clean == last_sequence_str:
                    patience_counter += 1
                else:
                    patience_counter = 0
                last_sequence_str = current_seq_str_clean

                if patience_counter >= patience:
                    print(f"\n[Early Stopping] Sequence converged for {patience} iterations at iteration {iteration}.")
                    break

            # 10. Console logging
            if iteration % log_interval == 0:
                history.append({
                    'iteration': iteration,
                    'kcat_pred': kcat_pred.item(),
                    'kcat_mean': kcat_pred.item(),
                    'kcat_std': kcat_std.item(),
                    'total_grad_norm': total_grad_norm,
                    'kcat_grad_norm': kcat_grad_norm if self.use_esmfold else None,
                    'struct_grad_norm': struct_grad_norm if self.use_esmfold else None,
                    'loss': loss.item(),
                    'penalty': penalty.item() if self.penalty_type else 0.0,
                    'struct_loss': struct_loss.item() if self.use_esmfold else 0.0,
                    'struct_reward': struct_score if self.use_esmfold else 0.0,
                    'temperature': temperature,
                    'sequence': current_iteration_indices.cpu().numpy()
                })
                penalty_str = f", penalty={penalty.item():.4f}" if self.penalty_type else ""
                struct_str = (
                    f", struct_loss={struct_loss.item():.4f}, struct_reward={struct_score:.2f}"
                    if self.use_esmfold else ""
                )
                grad_str = f", grad_norm={total_grad_norm:.6f}"
                if self.use_esmfold:
                    grad_str += f" (kcat={kcat_grad_norm:.6f}, struct={struct_grad_norm:.6f})"
                current_lr = self.get_current_lr(optimizer)
                lr_str = f", lr={current_lr:.6f}" if lr_scheduler != 'none' else ""
                std_print_str = f", std={kcat_std.item():.4f}" if self.kcat_loss_type == 'uncertainty_aware' else ""
                print(f"Iter {iteration}: kcat={kcat_pred.item():.4f}{std_print_str}, loss={loss.item():.4f}{penalty_str}{struct_str}, temp={temperature:.2f}{grad_str}{lr_str}")

        if csv_file:
            csv_file.close()

        # Return the final discrete sequence and full logits/probability matrices
        final_sequence = self.seq_logits.get_discrete_sequence()
        logits_matrix, probs_matrix, aa_mapping = self.seq_logits.get_full_logits_matrix_with_probs()

        return final_sequence, history, logits_matrix, probs_matrix, aa_mapping

    def _create_optimizer(self, optimizer_type='adam', lr=0.001, optimizer_kwargs=None):
        """
        Instantiate a PyTorch optimizer for the sequence logits.

        Args:
            optimizer_type: One of 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'.
            lr:             Learning rate.
            optimizer_kwargs: Extra keyword arguments forwarded to the optimizer.

        Returns:
            optimizer: PyTorch optimizer instance.
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        params = [self.seq_logits.logits_trainable]
        optimizer_type = optimizer_type.lower()

        if optimizer_type == 'adam':
            return torch.optim.Adam(params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == 'adagrad':
            return torch.optim.Adagrad(params, lr=lr, **optimizer_kwargs)
        else:
            raise ValueError(
                f"Unsupported optimizer type: '{optimizer_type}'. "
                f"Supported: adam, adamw, sgd, rmsprop, adagrad."
            )

    def _create_lr_scheduler(self, optimizer, lr_scheduler='none', num_iterations=1000,
                             lr_init=0.001, lr_min=1e-6):
        """
        Create a learning rate scheduler.

        Args:
            optimizer:      PyTorch optimizer instance.
            lr_scheduler:   Decay scheme ('none', 'linear', 'cosine').
            num_iterations: Total number of optimization iterations.
            lr_init:        Initial learning rate.
            lr_min:         Minimum learning rate.

        Returns:
            scheduler: LR scheduler instance, or None when lr_scheduler='none'.
        """
        import math

        if lr_scheduler == 'none':
            return None
        elif lr_scheduler == 'linear':
            def lr_lambda(step):
                if step >= num_iterations:
                    return lr_min / lr_init
                return max(lr_min / lr_init, 1.0 - step / num_iterations)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif lr_scheduler == 'cosine':
            def lr_lambda(step):
                if step >= num_iterations:
                    return lr_min / lr_init
                cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * step / num_iterations))
                return max(lr_min / lr_init, (lr_min + (lr_init - lr_min) * cosine_ratio) / lr_init)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(
                f"Unsupported LR scheduler: '{lr_scheduler}'. "
                f"Supported: none, linear, cosine."
            )

    def get_current_lr(self, optimizer):
        """Return the current learning rate from the optimizer."""
        return optimizer.param_groups[0]['lr']

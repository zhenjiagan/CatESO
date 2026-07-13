import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProteinSequenceLogits(nn.Module):
    """
    Trainable protein sequence logits.

    Core design: optimize only over the 20 standard amino acids.
    - Trainable parameter: P_canonical [L, 20]
    - Each forward builds a P_full [L, vocab_size] base matrix filled with -1e9,
      scatters P_canonical into the 20 canonical token positions,
      applies Gumbel-Softmax so illegal tokens have zero probability,
      and gradients flow only through those 20 positions to update P_canonical.
    """

    def __init__(self, seq_length, vocab_size=33, init_sequence=None, esm_alphabet=None,
                 fixed_positions=None):
        """
        Args:
            seq_length:      Sequence length L.
            vocab_size:      ESM-2 vocabulary size (33 residues + specials).
            init_sequence:   Optional initial sequence (string or token index list).
            esm_alphabet:    ESM-2 alphabet for encoding.
            fixed_positions: Dict mapping position → one-letter AA, e.g. {0: 'M', 5: 'A', 10: 'G'}.
        """
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.esm_alphabet = esm_alphabet
        self.fixed_positions = fixed_positions if fixed_positions is not None else {}

        # 20 standard amino acids
        self.standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.num_canonical = len(self.standard_aa)

        if esm_alphabet is not None:
            self.canonical_indices = torch.tensor([
                esm_alphabet.get_idx(aa) for aa in self.standard_aa
            ], dtype=torch.long)
            self.vocab_size = len(esm_alphabet)
        else:
            # Default ESM-2 token indices for the above AA order
            self.canonical_indices = torch.tensor(
                [5, 23, 13, 9, 18, 6, 21, 12, 15, 4,
                 20, 17, 14, 16, 10, 8, 11, 7, 22, 19],
                dtype=torch.long
            )
            self.vocab_size = vocab_size

        self._setup_mutable_and_fixed_indices()

        if init_sequence is not None:
            full_sequence_canonical_logits = self._sequence_to_canonical_logits(init_sequence)
        else:
            full_sequence_canonical_logits = torch.randn(seq_length, self.num_canonical) * 0.1

        if len(self.mutable_indices) > 0:
            self.logits_trainable = nn.Parameter(
                full_sequence_canonical_logits[self.mutable_indices, :]
            )
        else:
            raise ValueError(
                f"No mutable positions: all {seq_length} positions are fixed. "
                f"Sequence optimization requires at least one mutable position. "
                f"Check fixed_positions so not every site is fixed."
            )

        if len(self.fixed_indices) > 0:
            self.register_buffer(
                'logits_fixed',
                full_sequence_canonical_logits[self.fixed_indices, :]
            )
        else:
            self.register_buffer('logits_fixed', torch.empty(0, self.num_canonical))

    def _setup_mutable_and_fixed_indices(self):
        """Build sorted lists of mutable and fixed position indices."""
        all_positions = set(range(self.seq_length))
        fixed_pos_set = set(self.fixed_positions.keys())
        mutable_pos_set = all_positions - fixed_pos_set

        self.mutable_indices = sorted(list(mutable_pos_set))
        self.fixed_indices = sorted(list(fixed_pos_set))

        if len(self.fixed_indices) > 0:
            self.fixed_aa_canonical_indices = {}
            for pos, aa in self.fixed_positions.items():
                if aa in self.standard_aa:
                    canonical_idx = self.standard_aa.index(aa)
                    self.fixed_aa_canonical_indices[pos] = canonical_idx
                else:
                    raise ValueError(f"Fixed position {pos} has non-standard amino acid '{aa}'")

    def _sequence_to_canonical_logits(self, sequence):
        """
        Convert a discrete sequence to canonical logits [L, 20].

        Args:
            sequence: String (e.g. "ACDEFG...") or list of full-vocabulary token indices.

        Returns:
            logits_canonical: [L, 20] logits over the 20 standard amino acids.
        """
        logits_canonical = torch.zeros(self.seq_length, self.num_canonical)

        if isinstance(sequence, str):
            if self.esm_alphabet is None:
                raise ValueError("esm_alphabet is required to convert a string sequence to token indices")
            sequence = [self.esm_alphabet.get_idx(aa) for aa in sequence]

        alphabet_to_canonical = {
            alphabet_idx: canonical_idx
            for canonical_idx, alphabet_idx in enumerate(self.canonical_indices.tolist())
        }

        for i, alphabet_token_idx in enumerate(sequence):
            if i >= self.seq_length:
                break
            if alphabet_token_idx in alphabet_to_canonical:
                canonical_idx = alphabet_to_canonical[alphabet_token_idx]
                logits_canonical[i, canonical_idx] = 10.0
            else:
                raise ValueError(
                    f"Sequence contains non-standard amino acid token: {alphabet_token_idx}"
                )
            logits_canonical[i, :] += torch.randn(self.num_canonical) * 0.01

        return logits_canonical

    def _reconstruct_full_logits_canonical(self):
        """
        Rebuild full logits_canonical [L, 20].

        1. Zero template [L, 20].
        2. Scatter logits_trainable into mutable positions (gradients flow here).
        3. Scatter fixed positions with extreme logits (±1e9) so argmax is fixed AA.

        Returns:
            logits_canonical: [L, 20].
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
        Sample Gumbel(0, 1) using the current PyTorch RNG state.

        Args:
            shape:  Output shape.
            device: Target device.
            eps:    Numerical stability constant.

        Returns:
            Gumbel noise tensor.
        """
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, temperature=1.0, hard=False):
        """
        Sample a soft sequence with Gumbel-Softmax.

        Returns:
            soft_sequence: [L, vocab_size] soft one-hot.
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
        indices_expanded = canonical_indices.unsqueeze(0).expand(self.seq_length, -1)

        logits_full.scatter_(
            dim=1,
            index=indices_expanded,
            src=logits_canonical
        )

        gumbel_noise = self._sample_gumbel(logits_full.shape, device)
        y = logits_full + gumbel_noise
        soft_sequence = F.softmax(y / temperature, dim=-1)

        if hard:
            y_hard = torch.zeros_like(soft_sequence)
            y_hard.scatter_(1, soft_sequence.argmax(dim=-1, keepdim=True), 1.0)
            soft_sequence = (y_hard - soft_sequence).detach() + soft_sequence

        return soft_sequence

    def get_discrete_sequence(self):
        """
        Argmax discrete sequence over the full vocabulary (only canonical AAs reachable).

        Returns:
            discrete_sequence: [L] token indices.
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

        logits_full.scatter_(
            dim=1,
            index=indices_expanded,
            src=logits_canonical
        )

        discrete_sequence = torch.argmax(logits_full, dim=-1)
        return discrete_sequence

    def get_full_logits_matrix_with_probs(self):
        """
        Return full canonical logits and softmax probabilities for all positions.

        Returns:
            logits_canonical: [L, 20]
            probs_canonical:  [L, 20]
            aa_mapping:       list of 20 one-letter codes in canonical order
        """
        logits_canonical = self._reconstruct_full_logits_canonical()
        probs_canonical = F.softmax(logits_canonical, dim=-1)
        return logits_canonical, probs_canonical, self.standard_aa


class GumbelSoftSequence(nn.Module):
    """Gumbel-Softmax sampler with reproducible RNG and several temperature schedules."""

    def __init__(self, temperature_schedule="fixed", use_manual_gumbel=True):
        """
        Args:
            temperature_schedule: One of "fixed", "cosine", "linear", "exponential".
            use_manual_gumbel:    If True, use manual Gumbel sampling (seed-friendly).
        """
        super().__init__()
        self.temperature_schedule = temperature_schedule
        self.use_manual_gumbel = use_manual_gumbel

    def get_temperature(self, iteration, max_iterations, tau_init=1.0, tau_min=0.1):
        """
        Current temperature for this iteration.

        Args:
            iteration:        Current step.
            max_iterations:   Total steps.
            tau_init:         Initial temperature.
            tau_min:          Minimum temperature (annealing schedules).

        Returns:
            Scalar temperature.
        """
        if self.temperature_schedule == "fixed":
            return tau_init

        progress = min(iteration / max_iterations, 1.0) if max_iterations > 0 else 0.0

        if self.temperature_schedule == "cosine":
            import math
            return tau_min + 0.5 * (tau_init - tau_min) * (1 + math.cos(math.pi * progress))

        if self.temperature_schedule == "linear":
            return tau_init - (tau_init - tau_min) * progress

        if self.temperature_schedule == "exponential":
            return tau_init * ((tau_min / tau_init) ** progress)

        warnings.warn(
            f"Unknown temperature schedule '{self.temperature_schedule}'; using fixed temperature.",
            UserWarning,
            stacklevel=2,
        )
        return tau_init

    def sample_gumbel(self, shape, device, eps=1e-20):
        """Sample Gumbel(0, 1) noise."""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, eps=1e-20):
        """Manual Gumbel-Softmax sample."""
        gumbel_noise = self.sample_gumbel(logits.shape, logits.device, eps)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def forward(self, logits, temperature):
        """Gumbel-Softmax forward."""
        if self.use_manual_gumbel:
            return self.gumbel_softmax_sample(logits, temperature)
        return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)


class SequenceOptimizer:
    """Sequence optimization with optional ensemble models and penalty terms."""

    def __init__(self,
                 model,
                 esm_manager,
                 seq_length,
                 molecule_graph,
                 init_sequence=None,
                 fixed_positions=None,
                 device=None,
                 penalty_type=None,
                 penalty_lambda=0.0,
                 gumbel_schedule="fixed"):
        """
        Args:
            model:            Pretrained OmniESI model or list of models (ensemble).
            esm_manager:      ESM2Manager instance.
            molecule_graph:   Small-molecule graph (e.g. dgl.DGLGraph).
            seq_length:       Sequence length.
            init_sequence:    Initial sequence string (for logits and penalties).
            fixed_positions:  Dict {index: one-letter AA}.
            device:           Torch device.
            penalty_type:     'kl', 'l2', or None.
            penalty_lambda:   Penalty coefficient.
            gumbel_schedule:  'fixed', 'cosine', 'linear', or 'exponential'.
        """
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

        if device is None:
            self.device = next(self.models[0].parameters()).device
        else:
            self.device = device

        self.molecule_graph = molecule_graph.to(self.device)

        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False

        self.seq_logits = ProteinSequenceLogits(
            seq_length=seq_length,
            vocab_size=len(esm_manager.alphabet),
            init_sequence=init_sequence,
            esm_alphabet=esm_manager.alphabet,
            fixed_positions=fixed_positions
        ).to(self.device)

        self.token_embedding_weight = esm_manager.get_token_embedding().float().to(self.device)

        self.gumbel_sampler = GumbelSoftSequence(
            temperature_schedule=gumbel_schedule, use_manual_gumbel=True
        )

        if init_sequence is not None:
            self._init_original_onehot()
            if penalty_type == 'l2':
                self._init_original_embedding()

    def _init_original_onehot(self):
        """
        Smoothed target distribution over 20 canonical AAs for KL penalty.
        """
        self.original_onehot_canonical = torch.zeros(
            self.seq_length, self.seq_logits.num_canonical,
            device=self.device
        )

        for i, aa in enumerate(self.init_sequence):
            if aa in self.seq_logits.standard_aa:
                canonical_idx = self.seq_logits.standard_aa.index(aa)
                self.original_onehot_canonical[i, canonical_idx] = 1.0

        eps = 1e-8
        self.original_onehot_canonical = self.original_onehot_canonical + eps
        self.original_onehot_canonical = (
            self.original_onehot_canonical / self.original_onehot_canonical.sum(dim=-1, keepdim=True)
        )

    def _init_original_embedding(self):
        """Precompute ESM-2 embeddings of the initial sequence for L2/MSE penalty."""
        with torch.no_grad():
            original_onehot_full = torch.zeros(
                self.seq_length, len(self.esm_manager.alphabet),
                device=self.device
            )
            for i, aa in enumerate(self.init_sequence):
                idx = self.esm_manager.alphabet.get_idx(aa)
                original_onehot_full[i, idx] = 1.0

            self.original_feat = self._get_esm2_embedding(original_onehot_full)

    def _get_esm2_embedding(self, soft_sequence):
        """
        ESM-2 embeddings for sequence tokens only (no <cls>/<eos> in output).

        Args:
            soft_sequence: [L, vocab_size] soft one-hot.

        Returns:
            [L, 2560] embeddings.
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
            bos_onehot,
            soft_sequence,
            eos_onehot
        ], dim=0)

        soft_embeddings = torch.matmul(
            soft_sequence_with_special,
            self.token_embedding_weight
        )

        soft_embeddings = soft_embeddings.unsqueeze(0)
        esm_feat_with_special = self.esm_manager.forward_from_embeddings(soft_embeddings)
        esm_feat = esm_feat_with_special[:, 1:-1, :]

        return esm_feat.squeeze(0)

    def compute_kl_penalty(self, soft_sequence_canonical):
        """
        KL divergence penalty on mutable positions only.

        KL(P || Q) with P = softmax(current logits), Q = smoothed original one-hot.

        Args:
            soft_sequence_canonical: [L, 20] logits in canonical space.

        Returns:
            Scalar KL penalty.
        """
        if len(self.seq_logits.mutable_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        mutable_logits = soft_sequence_canonical[self.seq_logits.mutable_indices, :]
        mutable_onehot_smooth = self.original_onehot_canonical[self.seq_logits.mutable_indices, :]

        probs = F.softmax(mutable_logits, dim=-1)

        eps = 1e-8
        kl_div = probs * (torch.log(probs + eps) - torch.log(mutable_onehot_smooth))
        return kl_div.sum()

    def compute_l2_norm_penalty(self, soft_sequence):
        """
        MSE between current and original ESM-2 embeddings on mutable positions only.

        Args:
            soft_sequence: [L, vocab_size] soft one-hot (full vocab).

        Returns:
            Scalar MSE penalty.
        """
        if not hasattr(self, 'original_feat'):
            raise RuntimeError(
                "L2 penalty requires penalty_type='l2' at SequenceOptimizer init "
                "so original ESM-2 embeddings are precomputed."
            )

        if len(self.seq_logits.mutable_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        current_feat = self._get_esm2_embedding(soft_sequence)
        current_feat_mutable = current_feat[self.seq_logits.mutable_indices, :]
        original_feat_mutable = self.original_feat[self.seq_logits.mutable_indices, :]

        return F.mse_loss(current_feat_mutable, original_feat_mutable, reduction='mean')

    def soft_embedding_forward(self, soft_sequence):
        """
        Soft embedding → ESM-2 → OmniESI forward; ensemble returns mean score.

        Args:
            soft_sequence: (L, vocab_size).

        Returns:
            Predicted kcat (scalar batch dim).
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
            bos_onehot,
            soft_sequence,
            eos_onehot
        ], dim=0)

        soft_embeddings = torch.matmul(
            soft_sequence_with_special,
            self.token_embedding_weight
        )

        soft_embeddings = soft_embeddings.unsqueeze(0)
        esm_feat_with_special = self.esm_manager.forward_from_embeddings(soft_embeddings)
        esm_feat = esm_feat_with_special[:, 1:-1, :]

        v_p = esm_feat
        v_d = self.molecule_graph

        v_p_mask = torch.zeros(1, v_p.shape[1], dtype=torch.bool, device=self.device)
        v_d_mask = torch.zeros(1, v_d.num_nodes(), dtype=torch.bool, device=self.device)

        # drug_extractor may pop ndata['h']; restore after each forward
        node_feats_backup = v_d.ndata['h']

        if self.is_ensemble:
            scores = []
            for model in self.models:
                v_d.ndata['h'] = node_feats_backup
                v_d_out, v_p_out, fusion_out, score = model(v_d, v_p, v_d_mask, v_p_mask)
                scores.append(score)

            v_d.ndata['h'] = node_feats_backup
            all_scores = torch.stack(scores, dim=0)
            return all_scores.mean(dim=0)

        v_d.ndata['h'] = node_feats_backup
        v_d_out, v_p_out, fusion_out, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
        v_d.ndata['h'] = node_feats_backup
        return score

    def optimize(self,
                 num_iterations=1000,
                 lr=0.001,
                 log_interval=50,
                 tau=None,
                 tau_init=1.0,
                 tau_min=0.5,
                 optimizer_type='adam',
                 optimizer_kwargs=None,
                 ste_interval=None,
                 ste_mode='none'):
        """
        Optimize sequence logits (mutable positions only; fixed sites unchanged).

        Args:
            num_iterations:   Number of steps.
            lr:               Learning rate.
            log_interval:     Print progress every N steps.
            tau:              Deprecated; use tau_init.
            tau_init:         Initial Gumbel-Softmax temperature.
            tau_min:          Minimum temperature when annealing.
            optimizer_type:   'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'.
            optimizer_kwargs: Extra kwargs for the optimizer.
            ste_interval:     If > 0, apply STE / reinit every N steps (see ste_mode).
            ste_mode:         'none', 'hard', 'reinit_soft', or 'reinit_hard'.

        Returns:
            final_sequence, history, logits_matrix [L,20], probs_matrix [L,20], aa_mapping.
        """
        use_ste = ste_interval is not None and ste_interval > 0 and ste_mode != 'none'
        if use_ste:
            if ste_mode not in ['hard', 'reinit_soft', 'reinit_hard']:
                raise ValueError(
                    f"Unsupported STE mode '{ste_mode}'. "
                    f"Use 'none', 'hard', 'reinit_soft', or 'reinit_hard'."
                )

        optimizer = self._create_optimizer(
            optimizer_type=optimizer_type,
            lr=lr,
            optimizer_kwargs=optimizer_kwargs
        )
        history = []

        initial_tau = tau if tau is not None else tau_init

        for iteration in range(num_iterations):
            if use_ste and iteration > 0 and iteration % ste_interval == 0:
                if ste_mode == 'reinit_soft':
                    with torch.no_grad():
                        current_logits = self.seq_logits._reconstruct_full_logits_canonical()
                        discrete_seq = self.seq_logits.get_discrete_sequence()

                        target_logits = torch.zeros_like(current_logits)
                        alphabet_to_canonical = {
                            idx.item(): j
                            for j, idx in enumerate(self.seq_logits.canonical_indices)
                        }
                        for i, token_idx in enumerate(discrete_seq):
                            if token_idx.item() in alphabet_to_canonical:
                                canonical_idx = alphabet_to_canonical[token_idx.item()]
                                target_logits[i, canonical_idx] = 10.0

                        alpha = 0.5
                        new_logits = alpha * target_logits + (1 - alpha) * current_logits
                        self.seq_logits.logits_trainable.copy_(
                            new_logits[self.seq_logits.mutable_indices, :]
                        )

                elif ste_mode == 'reinit_hard':
                    with torch.no_grad():
                        discrete_seq = self.seq_logits.get_discrete_sequence()
                        new_logits_full = self.seq_logits._sequence_to_canonical_logits(
                            discrete_seq.cpu().numpy().tolist()
                        ).to(self.device)
                        self.seq_logits.logits_trainable.copy_(
                            new_logits_full[self.seq_logits.mutable_indices, :]
                        )

                    optimizer.state.clear()

            temperature = self.gumbel_sampler.get_temperature(
                iteration, num_iterations, tau_init=initial_tau, tau_min=tau_min
            )

            use_hard_now = (
                use_ste and
                ste_mode == 'hard' and
                iteration > 0 and
                iteration % ste_interval == 0
            )

            soft_sequence = self.seq_logits(temperature=temperature, hard=use_hard_now)

            kcat_pred = self.soft_embedding_forward(soft_sequence)
            loss = -kcat_pred

            penalty = torch.tensor(0.0, device=self.device)
            if self.penalty_type is not None and self.penalty_lambda > 0:
                if self.penalty_type == 'kl':
                    logits_canonical = self.seq_logits._reconstruct_full_logits_canonical()
                    penalty = self.compute_kl_penalty(logits_canonical)
                    loss = loss + self.penalty_lambda * penalty
                elif self.penalty_type == 'l2':
                    penalty = self.compute_l2_norm_penalty(soft_sequence)
                    loss = loss + self.penalty_lambda * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                discrete_seq = self.seq_logits.get_discrete_sequence()
                history.append({
                    'iteration': iteration,
                    'kcat_pred': kcat_pred.item(),
                    'loss': loss.item(),
                    'penalty': penalty.item() if self.penalty_type else 0.0,
                    'temperature': temperature,
                    'sequence': discrete_seq.cpu().numpy()
                })
                penalty_str = f", penalty={penalty.item():.4f}" if self.penalty_type else ""
                print(
                    f"Iter {iteration}: kcat={kcat_pred.item():.4f}, "
                    f"loss={loss.item():.4f}{penalty_str}, temp={temperature:.2f}"
                )

        final_sequence = self.seq_logits.get_discrete_sequence()
        logits_matrix, probs_matrix, aa_mapping = self.seq_logits.get_full_logits_matrix_with_probs()

        return final_sequence, history, logits_matrix, probs_matrix, aa_mapping

    def _create_optimizer(self, optimizer_type='adam', lr=0.001, optimizer_kwargs=None):
        """
        Build a PyTorch optimizer over trainable sequence logits.

        Args:
            optimizer_type:   'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'.
            lr:               Learning rate.
            optimizer_kwargs: Extra optimizer kwargs.

        Returns:
            Optimizer instance.
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        params = [self.seq_logits.logits_trainable]
        optimizer_type = optimizer_type.lower()

        if optimizer_type == 'adam':
            return torch.optim.Adam(params, lr=lr, **optimizer_kwargs)
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(params, lr=lr, **optimizer_kwargs)
        if optimizer_type == 'sgd':
            return torch.optim.SGD(params, lr=lr, **optimizer_kwargs)
        if optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, **optimizer_kwargs)
        if optimizer_type == 'adagrad':
            return torch.optim.Adagrad(params, lr=lr, **optimizer_kwargs)

        raise ValueError(
            f"Unsupported optimizer type '{optimizer_type}'. "
            f"Use adam, adamw, sgd, rmsprop, or adagrad."
        )

import torch
import torch.utils.checkpoint
import esm
from esm import pretrained
from pathlib import Path
import os

class ESM2Manager:
    """Manages the ESM-2 model for both offline and online usage modes."""

    def __init__(self, model_name="esm2_t36_3B_UR50D", model_path=None):
        """
        Args:
            model_name: Model name used for downloading from the hub.
            model_path: Local model path; if provided, loads from disk instead of the hub.
        """
        if model_path is not None:
            print(f"Loading ESM-2 model from local path: {model_path}")
            self.model, self.alphabet = pretrained.load_model_and_alphabet_local(model_path)
        else:
            # Prefer TORCH_HOME for cache lookup; fall back to the default ~/.cache path.
            torch_home = os.environ.get('TORCH_HOME', None)
            if torch_home:
                cache_path = Path(torch_home) / "hub" / "checkpoints" / f"{model_name}.pt"
            else:
                cache_path = Path.home() / ".cache/torch/hub/checkpoints" / f"{model_name}.pt"

            if cache_path.exists():
                print(f"Loading ESM-2 model from cache: {cache_path}")
                self.model, self.alphabet = pretrained.load_model_and_alphabet_local(str(cache_path))
            else:
                print(f"Downloading ESM-2 model from hub: {model_name}")
                self.model, self.alphabet = pretrained.load_model_and_alphabet_hub(model_name)

        self.model.eval()

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features_offline(self, protein_path):
        """Mode 1: load pre-computed features from disk (used during training)."""
        return torch.load(protein_path)  # (seq_len, 2560)

    def extract_features_online(self, sequence):
        """Mode 2: extract features on-the-fly (used during sequence optimization)."""
        batch_converter = self.alphabet.get_batch_converter()
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[36])
        return results["representations"][36][0, 1:-1, :]  # (seq_len, 2560)

    def get_token_embedding(self):
        """Return the token embedding weight matrix (used for soft-embedding forward pass)."""
        return self.model.embed_tokens.weight  # (33, 2560)

    @staticmethod
    def _layer_forward(layer, x, padding_mask):
        """Single Transformer layer forward, used as the unit for gradient checkpointing."""
        x, _ = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=False)
        return x

    def forward_from_embeddings(self, soft_embeddings, attention_mask=None):
        """
        Run a forward pass starting directly from soft embeddings, bypassing embed_tokens.

        Uses gradient checkpointing to trade a small amount of extra compute for a large
        reduction in intermediate activation memory.

        Args:
            soft_embeddings: (batch, tokens_len=seq_len+2, embed_dim=2560) soft embedding tensor.
            attention_mask:  (batch, seq_len) optional attention mask.

        Returns:
            representations: (batch, seq_len, embed_dim) last-layer hidden states.
        """
        x = soft_embeddings * self.model.embed_scale
        batch_size, tokens_len, embed_dim = x.shape

        # Cast to the model's dtype to support shared BF16 models
        model_dtype = next(self.model.parameters()).dtype
        original_dtype = x.dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)

        # Build padding mask (all-False = no padding) if not provided
        if attention_mask is None:
            padding_mask = torch.zeros(batch_size, tokens_len, dtype=torch.bool, device=x.device)
        else:
            padding_mask = attention_mask

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) → (T, B, E): ESM-2 uses sequence-first layout internally
        x = x.transpose(0, 1)

        # Pass None to the attention layers when there is no actual padding
        if not padding_mask.any():
            padding_mask = None

        # Forward through Transformer layers with gradient checkpointing
        for layer_idx, layer in enumerate(self.model.layers):
            x = torch.utils.checkpoint.checkpoint(
                self._layer_forward, layer, x, padding_mask
            )

        # Final layer norm
        x = self.model.emb_layer_norm_after(x)

        # (T, B, E) → (B, T, E)
        x = x.transpose(0, 1)

        # Restore original dtype (FP32) for compatibility with downstream modules
        if x.dtype != original_dtype:
            x = x.to(original_dtype)

        return x

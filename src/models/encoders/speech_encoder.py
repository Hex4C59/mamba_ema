"""Speech encoder using pre-trained WavLM/Wav2Vec2 models with caching."""

from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class SpeechEncoder(nn.Module):
    """Speech encoder with pooling layer and feature caching.

    Uses pre-trained WavLM or Wav2Vec2 to extract utterance-level features.
    Supports multi-layer feature extraction with learnable weighted fusion.

    Args:
        model_name: HuggingFace model name
        pooling: Pooling method ("mean" or "attention")
        freeze: If True, freeze encoder weights
        d_output: Output dimension (default: 768)
        cache_dir: Directory to cache extracted features (default: /tmp/wavlm_cache)
        use_cache: If True, use cached features when available
        extract_layers: List of layer indices to extract (e.g., [6, 12, 18, 24])
                       If None, use only final layer
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        pooling: str = "mean",
        freeze: bool = True,
        d_output: int = 768,
        cache_dir: str = "/tmp/wavlm_cache",
        use_cache: bool = True,
        extract_layers: List[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.d_output = d_output
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.extract_layers = extract_layers

        # Create cache directory if using cache
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load pre-trained model
        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Enable hidden states output if using multi-layer extraction
        if extract_layers is not None:
            self.model.config.output_hidden_states = True
            self.num_layers = len(extract_layers)
            # Learnable layer weights (initialized uniformly)
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        else:
            self.num_layers = 1

        # Freeze/unfreeze
        if freeze:
            self.freeze()

        # Attention pooling layer
        if pooling == "attention":
            self.attention = nn.Linear(d_output, 1)

    def freeze(self) -> None:
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self, waveforms: List[torch.Tensor], names: List[str] | None = None
    ) -> torch.Tensor:
        """Extract utterance-level features with caching.

        Args:
            waveforms: List of [T_i] tensors (variable length)
            names: List of sample names for cache lookup (optional)

        Returns:
            Tensor [B, d_output]
        """
        batch_size = len(waveforms)
        device = next(self.model.parameters()).device

        features = []
        for i, wf in enumerate(waveforms):
            # Try to load from cache if enabled and name provided
            cache_path = None
            if self.use_cache and names is not None:
                cache_path = self.cache_dir / f"{names[i]}.pt"

                if cache_path.exists():
                    # Load cached feature
                    try:
                        cached_feat = torch.load(
                            cache_path, map_location=device, weights_only=True
                        )
                        features.append(cached_feat)
                        continue  # Skip extraction for this sample
                    except Exception as e:
                        # If loading fails, extract and overwrite cache
                        print(f"Warning: Failed to load cache {cache_path}: {e}")

            # Extract features if not cached
            wf = wf.to(device)
            with torch.set_grad_enabled(self.training and not self._is_frozen()):
                outputs = self.model(wf.unsqueeze(0))

                # Multi-layer feature extraction with weighted fusion
                if self.extract_layers is not None:
                    hidden_states = outputs.hidden_states  # Tuple of [1, T, d_output]
                    # Extract specified layers
                    layer_features = [hidden_states[layer_idx] for layer_idx in self.extract_layers]
                    # Weighted fusion: sum(weight_i * layer_i)
                    weights = torch.softmax(self.layer_weights, dim=0)  # Normalize weights
                    hidden = sum(w * layer for w, layer in zip(weights, layer_features))  # [1, T, d_output]
                else:
                    hidden = outputs.last_hidden_state  # [1, T, d_output]

            # Pooling
            if self.pooling == "mean":
                pooled = hidden.mean(dim=1).squeeze(0)  # [d_output]
            elif self.pooling == "attention":
                # Attention pooling
                attn_weights = torch.softmax(
                    self.attention(hidden).squeeze(-1), dim=1
                )  # [1, T]
                pooled = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1).squeeze(
                    0
                )  # [d_output]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            features.append(pooled)

            # Save to cache if enabled
            if self.use_cache and cache_path is not None:
                try:
                    torch.save(pooled.cpu(), cache_path)
                except Exception as e:
                    print(f"Warning: Failed to save cache {cache_path}: {e}")

        # Stack batch
        return torch.stack(features)  # [B, d_output]

    def _is_frozen(self) -> bool:
        """Check if model is frozen."""
        return not next(self.model.parameters()).requires_grad

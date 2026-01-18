"""Extract WavLM and ECAPA-TDNN features for offline training.

Usage:
    uv run python scripts/extract_features.py \
        --label_file data/labels/IEMOCAP/iemocap_label.csv \
        --audio_root /tmp/IEMOCAP_full_release \
        --output_dir data/features/IEMOCAP \
        --features wavlm ecapa \
        --wavlm_model pretrained_model/wavlm-large \
        --wavlm_layers 6 12 18 24
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features for offline training")
    parser.add_argument("--label_file", type=str, required=True, help="Label CSV file")
    parser.add_argument("--audio_root", type=str, required=True, help="Audio root dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir")
    parser.add_argument("--features", nargs="+", default=["wavlm", "ecapa"],
                        choices=["wavlm", "ecapa"], help="Features to extract")
    parser.add_argument("--wavlm_model", type=str, default="pretrained_model/wavlm-large")
    parser.add_argument("--wavlm_layers", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 for safety)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing files")
    return parser.parse_args()


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio and resample to target sample rate."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [T]


class WavLMExtractor:
    """Extract WavLM features."""

    def __init__(self, model_path: str, layers: list[int], device: str):
        self.device = device
        self.layers = layers
        print(f"Loading WavLM from {model_path}...")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.config.output_hidden_states = True
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> dict:
        """Extract features from waveform.

        Returns:
            dict with:
                - 'features': [num_layers, T, D] multi-layer features
                - 'length': int
                - 'layers': list of layer indices
        """
        waveform = waveform.to(self.device).unsqueeze(0)  # [1, T]
        outputs = self.model(waveform)

        # Extract specified layers, keep them separate for learnable fusion
        hidden_states = outputs.hidden_states  # tuple of [1, T', D]
        layer_features = [hidden_states[i].squeeze(0) for i in self.layers]  # list of [T', D]
        features = torch.stack(layer_features, dim=0)  # [num_layers, T', D]

        return {
            "features": features.cpu(),
            "length": features.shape[1],
            "layers": self.layers,
        }


class ECAPAExtractor:
    """Extract ECAPA-TDNN speaker embeddings."""

    def __init__(self, device: str):
        from speechbrain.inference import EncoderClassifier
        print("Loading ECAPA-TDNN...")
        device_str = "cuda" if "cuda" in device else "cpu"
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_model/spkrec-ecapa-voxceleb",
            run_opts={"device": device_str},
        )
        self.ecapa.eval()
        self.device = device

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding [192]."""
        waveform = waveform.to(self.device).unsqueeze(0)  # [1, T]
        wav_lens = torch.tensor([1.0], device=self.device)
        embedding = self.ecapa.encode_batch(waveform, wav_lens)
        while embedding.dim() > 1:
            embedding = embedding.squeeze(0)
        return embedding.cpu()  # [192]


def main():
    args = parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load label file
    df = pd.read_csv(args.label_file)
    audio_root = Path(args.audio_root)
    output_dir = Path(args.output_dir)

    # Replace audio path prefix
    if "audio_path" in df.columns:
        # Extract relative path after known prefixes
        prefixes = ["/tmp/IEMOCAP_full_release", "/tmp/CCSEMO"]
        for prefix in prefixes:
            df["audio_path"] = df["audio_path"].str.replace(prefix, str(audio_root), regex=False)

    # Create output directories
    feature_dirs = {}
    for feat in args.features:
        feat_dir = output_dir / feat
        feat_dir.mkdir(parents=True, exist_ok=True)
        feature_dirs[feat] = feat_dir

    # Initialize extractors
    extractors = {}
    if "wavlm" in args.features:
        extractors["wavlm"] = WavLMExtractor(args.wavlm_model, args.wavlm_layers, device)
    if "ecapa" in args.features:
        extractors["ecapa"] = ECAPAExtractor(device)

    # Extract features
    stats = {"total": len(df), "wavlm": 0, "ecapa": 0, "skipped": 0, "failed": 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        name = row["name"]
        audio_path = row["audio_path"]

        try:
            waveform = load_audio(audio_path, args.sample_rate)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            stats["failed"] += 1
            continue

        for feat_name, extractor in extractors.items():
            output_path = feature_dirs[feat_name] / f"{name}.pt"

            if args.skip_existing and output_path.exists():
                stats["skipped"] += 1
                continue

            try:
                if feat_name == "wavlm":
                    result = extractor.extract(waveform)
                    torch.save(result, output_path)
                elif feat_name == "ecapa":
                    embedding = extractor.extract(waveform)
                    torch.save(embedding, output_path)
                stats[feat_name] += 1
            except Exception as e:
                print(f"Failed to extract {feat_name} for {name}: {e}")
                stats["failed"] += 1

    # Save metadata
    metadata = {
        "label_file": str(args.label_file),
        "audio_root": str(args.audio_root),
        "features": args.features,
        "wavlm_model": args.wavlm_model if "wavlm" in args.features else None,
        "wavlm_layers": args.wavlm_layers if "wavlm" in args.features else None,
        "sample_rate": args.sample_rate,
        "stats": stats,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Stats: {stats}")
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()

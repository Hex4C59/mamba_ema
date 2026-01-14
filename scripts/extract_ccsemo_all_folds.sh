#!/bin/bash
# Extract eGeMAPS features for CCSEMO dataset
# Since 5-fold CV only changes train/val/test split, we only need to extract once

# Configuration
AUDIO_ROOT="/tmp/CCSEMO/audio"
LABEL_DIR="data/labels/CCSEMO/5fold"
OUTPUT_DIR="data/features/CCSEMO/egemaps"

echo "Extracting eGeMAPS features for CCSEMO..."
echo "Audio root: $AUDIO_ROOT"
echo "Label dir: $LABEL_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if audio directory exists
if [ ! -d "$AUDIO_ROOT" ]; then
    echo "Error: Audio directory not found: $AUDIO_ROOT"
    echo "Please update AUDIO_ROOT in this script or create the directory."
    exit 1
fi

# Extract features once for all unique audio files
uv run python scripts/extract_egemaps_ccsemo.py \
    --label_dir "$LABEL_DIR" \
    --audio_root "$AUDIO_ROOT" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Feature extraction complete!"
echo "========================================"
echo "Features saved to: $OUTPUT_DIR"
echo ""
echo "You can now train with any fold:"
echo "  uv run python src/train.py --config configs/ccsemo.yaml --fold 1 --gpu 0"


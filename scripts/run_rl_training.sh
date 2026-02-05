#!/bin/bash
#
# Q-Table Training Script for GPUCompress
#
# This script trains the Q-Table for automatic compression algorithm selection.
#
# Usage:
#   ./scripts/run_rl_training.sh [options]
#
# Options:
#   --data-dir DIR    Training data directory (required)
#   --epochs N        Number of training epochs (default: 100)
#   --preset NAME     Reward preset (balanced, max_ratio, max_speed, etc.)
#   --error-bound E   Quantization error bound (default: 0.001)
#

set -e

# Default values
DATA_DIR=""
OUTPUT_DIR="rl/models"
EPOCHS=100
PRESET="balanced"
ERROR_BOUND=0.001
GPU_COMPRESS="./build/gpu_compress"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir|-d)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --preset|-p)
            PRESET="$2"
            shift 2
            ;;
        --error-bound)
            ERROR_BOUND="$2"
            shift 2
            ;;
        --gpu-compress)
            GPU_COMPRESS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Q-Table Training for GPUCompress"
            echo ""
            echo "Usage: $0 --data-dir DIR [options]"
            echo ""
            echo "Options:"
            echo "  --data-dir, -d DIR    Training data directory (required)"
            echo "  --output-dir, -o DIR  Output directory (default: rl/models)"
            echo "  --epochs, -e N        Training epochs (default: 100)"
            echo "  --preset, -p NAME     Reward preset (default: balanced)"
            echo "                        Options: balanced, max_ratio, max_speed,"
            echo "                                 max_quality, storage, streaming"
            echo "  --error-bound E       Quantization error bound (default: 0.001)"
            echo "  --gpu-compress PATH   Path to gpu_compress binary"
            echo ""
            echo "Example:"
            echo "  $0 --data-dir datasets/training --epochs 200 --preset balanced"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$DATA_DIR" ]; then
    echo "Error: --data-dir is required"
    echo "Run with --help for usage"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -x "$GPU_COMPRESS" ]; then
    echo "Error: gpu_compress not found or not executable: $GPU_COMPRESS"
    echo "Build first with: mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# Print configuration
echo "=============================================="
echo "Q-Table Training Configuration"
echo "=============================================="
echo "Data directory:  $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs:          $EPOCHS"
echo "Reward preset:   $PRESET"
echo "Error bound:     $ERROR_BOUND"
echo "GPU compress:    $GPU_COMPRESS"
echo "=============================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Run training
python3 -m rl.trainer \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --preset "$PRESET" \
    --error-bound "$ERROR_BOUND" \
    --gpu-compress "$GPU_COMPRESS"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  JSON model: $OUTPUT_DIR/qtable.json"
echo "  Binary model: $OUTPUT_DIR/qtable.bin"
echo ""
echo "To use the trained model:"
echo "  1. Copy qtable.bin to a known location"
echo "  2. Initialize library: gpucompress_init(\"/path/to/qtable.bin\")"
echo "  3. Use GPUCOMPRESS_ALGO_AUTO for automatic selection"
echo ""

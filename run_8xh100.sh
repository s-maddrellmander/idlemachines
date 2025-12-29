#!/bin/bash
# run_8xh100.sh
# Launch script for 8xH100 training
#
# Usage:
#   ./run_8xh100.sh fp8      # FP8 training
#   ./run_8xh100.sh bf16     # BF16 baseline

set -e

MODE=${1:-fp8}
MODEL_SIZE=${2:-1.5B}
STEPS=${3:-15000}

# ============================================================================
# Optimal settings for 8xH100 (80GB each)
# ============================================================================
#
# 1.5B model memory estimate:
#   - Model weights (BF16): ~3GB
#   - Gradients: ~3GB  
#   - Optimizer states (FP32): ~18GB
#   - Activations (batch_size dependent): ~10-20GB
#   Total: ~35-45GB per GPU with batch_size=32
#
# With 8 GPUs:
#   - batch_size=32 per GPU × 8 GPUs = 256 sequences per step
#   - 256 × 1024 tokens = 262k tokens per step
#   - With grad_accum=2: 524k tokens per step
#
# Chinchilla-optimal for 1.5B: ~30B tokens
# 30B / 524k = ~57k steps
# For a shorter scaling law run: 10k steps = ~5.2B tokens
# ============================================================================

BATCH_SIZE=32          # Per-GPU micro batch size
GRAD_ACCUM=8           # Gradient accumulation steps
SEQ_LEN=1024           # Sequence length

# Effective batch size: 32 × 2 × 8 = 512 sequences = 524k tokens/step

# Learning rate (scale with sqrt of batch size from baseline)
# Baseline: 6e-4 for batch=480k tokens
# Our batch: 524k tokens, so roughly same LR
LR=6e-4
MIN_LR=6e-5
WARMUP_FRAC=0.02

# Validation
VAL_EVERY=500
VAL_BATCHES=50
LOG_EVERY=10

# Paths
TRAIN_DATA="data/pretrain/train_*.bin"
VAL_DATA="data/pretrain/val_*.bin"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# Mode-specific settings
# ============================================================================

if [ "$MODE" == "fp8" ]; then
    echo "=========================================="
    echo "FP8 Training on 8xH100"
    echo "=========================================="
    FP8_FLAG=""
    FP8_RECIPE="tensorwise"
    LOG_FILE="logs/train_fp8_${MODEL_SIZE}_8xh100_${TIMESTAMP}.csv"
    CHECKPOINT="checkpoints/fp8_${MODEL_SIZE}_8xh100.pt"
elif [ "$MODE" == "bf16" ]; then
    echo "=========================================="
    echo "BF16 Baseline on 8xH100"
    echo "=========================================="
    FP8_FLAG="--no-fp8"
    FP8_RECIPE="tensorwise"  # ignored
    LOG_FILE="logs/train_bf16_${MODEL_SIZE}_8xh100_${TIMESTAMP}.csv"
    CHECKPOINT="checkpoints/bf16_${MODEL_SIZE}_8xh100.pt"
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [fp8|bf16] [model_size] [steps]"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints

# ============================================================================
# Print configuration
# ============================================================================

echo ""
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  Mode: $MODE"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE per GPU"
echo "  Grad accum: $GRAD_ACCUM"
echo "  World size: 8 GPUs"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM * 8)) sequences"
echo "  Tokens/step: $((BATCH_SIZE * GRAD_ACCUM * 8 * SEQ_LEN))"
echo "  Total tokens: ~$((STEPS * BATCH_SIZE * GRAD_ACCUM * 8 * SEQ_LEN / 1000000000))B"
echo "  Log file: $LOG_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo ""

# ============================================================================
# Launch training
# ============================================================================

# Set environment variables for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# Optional: Pin memory for faster data loading
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_gpt_fp8.py \
    --model-size $MODEL_SIZE \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --seq-len $SEQ_LEN \
    --lr $LR \
    --min-lr $MIN_LR \
    --warmup-frac $WARMUP_FRAC \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --val-every $VAL_EVERY \
    --val-batches $VAL_BATCHES \
    --log-every $LOG_EVERY \
    --log-file $LOG_FILE \
    --checkpoint-path $CHECKPOINT \
    --fp8-recipe $FP8_RECIPE \
    $FP8_FLAG

echo ""
echo "Training complete!"
echo "Log: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT"

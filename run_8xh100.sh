#!/bin/bash
# run_8xh200.sh
# Launch script for 8xH200 training - OPTIMIZED
#
# Usage:
#   ./run_8xh200.sh fp8 125M chinchilla    # Chinchilla-optimal 125M (~2.5B tokens)
#   ./run_8xh200.sh fp8 1.5B chinchilla    # Chinchilla-optimal 1.5B (~30B tokens)  
#   ./run_8xh200.sh bf16 125M 1500         # Custom step count
#   ./run_8xh200.sh fp8 1.5B 15000         # Full 1.5B run

set -e

MODE=${1:-fp8}
MODEL_SIZE=${2:-1.5B}
STEPS_OR_CHINCHILLA=${3:-chinchilla}

# ============================================================================
# H200-SPECIFIC OPTIMIZATIONS (141GB HBM3e per GPU, 4.8 TB/s bandwidth)
# ============================================================================
#
# H200 has ~40% more memory and ~43% more bandwidth than H100
# We can use MUCH larger batch sizes, especially for smaller models
#
# Memory estimates for different model sizes:
#   125M:  ~5GB total  → batch_size=64 easily fits
#   1.5B:  ~45GB total → batch_size=48 fits comfortably
#   3B:    ~90GB total → batch_size=32 fits
#   7B:    ~200GB total → batch_size=16 with grad_accum
# ============================================================================

# Model-specific optimal settings for H200
case $MODEL_SIZE in
    "125M")
        BATCH_SIZE=128          # H200 can handle much larger batches for small models
        GRAD_ACCUM=2           # Reduce accum since batch is larger
        LR=6e-4
        MIN_LR=6e-5
        CHINCHILLA_TOKENS=2500000000   # 2.5B tokens (20x params)
        ;;
    "250M")
        BATCH_SIZE=64
        GRAD_ACCUM=4
        LR=5e-4
        MIN_LR=5e-5
        CHINCHILLA_TOKENS=5000000000   # 5B tokens
        ;;
    "350M")
        BATCH_SIZE=48
        GRAD_ACCUM=4
        LR=4e-4
        MIN_LR=4e-5
        CHINCHILLA_TOKENS=7000000000   # 7B tokens
        ;;
    "760M")
        BATCH_SIZE=48
        GRAD_ACCUM=4
        LR=3e-4
        MIN_LR=3e-5
        CHINCHILLA_TOKENS=15000000000  # 15B tokens
        ;;
    "1.5B")
        BATCH_SIZE=64          # H200 can do 48 vs H100's 32
        GRAD_ACCUM=4           # Less accum needed with larger batch
        LR=3e-4
        MIN_LR=3e-5
        CHINCHILLA_TOKENS=30000000000  # 30B tokens
        ;;
    "3B")
        BATCH_SIZE=32
        GRAD_ACCUM=8
        LR=2e-4
        MIN_LR=2e-5
        CHINCHILLA_TOKENS=60000000000  # 60B tokens
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Available: 125M, 250M, 350M, 760M, 1.5B, 3B"
        exit 1
        ;;
esac

SEQ_LEN=1024
WORLD_SIZE=8

# Calculate tokens per step
TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM * WORLD_SIZE * SEQ_LEN))

# Calculate steps
if [ "$STEPS_OR_CHINCHILLA" == "chinchilla" ]; then
    STEPS=$((CHINCHILLA_TOKENS / TOKENS_PER_STEP))
    echo "=========================================="
    echo "CHINCHILLA-OPTIMAL ${MODEL_SIZE} on 8xH200"
    echo "=========================================="
else
    STEPS=$STEPS_OR_CHINCHILLA
    echo "=========================================="
    echo "${MODEL_SIZE} Training on 8xH200"
    echo "=========================================="
fi

WARMUP_FRAC=0.02

# Validation - more frequent for shorter runs
VAL_EVERY=500
LOG_EVERY=10
VAL_BATCHES=50

# Paths
TRAIN_DATA="data/pretrain/train_*.bin"
VAL_DATA="data/pretrain/val_*.bin"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="experiments/${MODEL_SIZE}_${MODE}_${TIMESTAMP}"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

# ============================================================================
# Mode-specific settings
# ============================================================================

if [ "$MODE" == "fp8" ]; then
    FP8_FLAG=""
    FP8_RECIPE="tensorwise"
    LOG_FILE="${EXPERIMENT_DIR}/train_fp8_${MODEL_SIZE}.csv"
    CHECKPOINT="${EXPERIMENT_DIR}/checkpoint_fp8_${MODEL_SIZE}.pt"
elif [ "$MODE" == "bf16" ]; then
    FP8_FLAG="--no-fp8"
    FP8_RECIPE="tensorwise"
    LOG_FILE="${EXPERIMENT_DIR}/train_bf16_${MODEL_SIZE}.csv"
    CHECKPOINT="${EXPERIMENT_DIR}/checkpoint_bf16_${MODEL_SIZE}.pt"
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [fp8|bf16] [model_size] [steps|chinchilla]"
    exit 1
fi

# ============================================================================
# Print configuration
# ============================================================================

TOTAL_TOKENS=$((STEPS * TOKENS_PER_STEP))
TOTAL_TOKENS_B=$(echo "scale=2; $TOTAL_TOKENS / 1000000000" | bc)

echo ""
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  Mode: $MODE"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE per GPU"
echo "  Grad accum: $GRAD_ACCUM"
echo "  World size: $WORLD_SIZE GPUs"
echo "  Sequence length: $SEQ_LEN"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM * WORLD_SIZE)) sequences"
echo "  Tokens/step: $TOKENS_PER_STEP"
echo "  Total tokens: ${TOTAL_TOKENS_B}B"
echo "  Learning rate: $LR → $MIN_LR"
echo "  Warmup: ${WARMUP_FRAC} of training"
echo "  Val every: $VAL_EVERY steps"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo ""

# Save config to experiment dir
cat > "${EXPERIMENT_DIR}/config.txt" << EOF
Model: $MODEL_SIZE
Mode: $MODE
Steps: $STEPS
Batch size per GPU: $BATCH_SIZE
Gradient accumulation: $GRAD_ACCUM
World size: $WORLD_SIZE
Sequence length: $SEQ_LEN
Effective batch: $((BATCH_SIZE * GRAD_ACCUM * WORLD_SIZE))
Tokens per step: $TOKENS_PER_STEP
Total tokens: ${TOTAL_TOKENS_B}B
Learning rate: $LR → $MIN_LR
Warmup fraction: $WARMUP_FRAC
Validation every: $VAL_EVERY
Started: $(date)
EOF

# ============================================================================
# Launch training
# ============================================================================

# Environment variables for optimal performance on H200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TF32 should be enabled by default, but ensure it
export NVIDIA_TF32_OVERRIDE=1

torchrun \
    --nproc_per_node=$WORLD_SIZE \
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
    $FP8_FLAG \
    2>&1 | tee "${EXPERIMENT_DIR}/training.log"

# ============================================================================
# Post-training
# ============================================================================

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Log: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "Training log: ${EXPERIMENT_DIR}/training.log"

# Append completion time
echo "Completed: $(date)" >> "${EXPERIMENT_DIR}/config.txt"

#!/bin/bash
#
# FP8 Training Experiment Launcher
# =================================
#
# Run multiple FP8 vs BF16 experiments sequentially.
# Designed for overnight/multi-day runs on DGX Spark.
#
# Usage:
#   ./run_experiments.sh                    # Run default experiments
#   ./run_experiments.sh --quick            # Quick test (100 steps each)
#   ./run_experiments.sh --500m             # 500M token runs
#   ./run_experiments.sh --full             # Full 500M runs on multiple sizes
#   ./run_experiments.sh --custom 1.5B 5000 # Custom: model_size, steps
#

set -e  # Exit on error

# Default settings
SCRIPT="train_gpt_fp8.py"
DATA_DIR="data/fineweb10B"
OUTPUT_DIR="experiments"
BATCH_SIZE=32
SEQ_LEN=1024
VAL_EVERY=500
LOG_EVERY=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}"
echo "============================================================"
echo "  FP8 Training Experiment Launcher"
echo "  DGX Spark / GB10"
echo "============================================================"
echo -e "${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run a single experiment
run_experiment() {
    local model_size=$1
    local steps=$2
    local mode=$3  # "bf16", "fp8_tensorwise", "fp8_rowwise"
    local batch_size=${4:-$BATCH_SIZE}
    
    echo -e "${YELLOW}"
    echo "------------------------------------------------------------"
    echo "  Experiment: ${model_size} / ${mode} / ${steps} steps"
    echo "------------------------------------------------------------"
    echo -e "${NC}"
    
    local log_file="${OUTPUT_DIR}/train_${mode}_${model_size}_${steps}steps_${TIMESTAMP}.csv"
    
    # Build command
    local cmd="python $SCRIPT \
        --model-size $model_size \
        --steps $steps \
        --batch-size $batch_size \
        --seq-len $SEQ_LEN \
        --val-every $VAL_EVERY \
        --log-every $LOG_EVERY \
        --train-data '${DATA_DIR}/fineweb_train_*.bin' \
        --val-data '${DATA_DIR}/fineweb_val_*.bin' \
        --log-file $log_file"
    
    if [ "$mode" == "bf16" ]; then
        cmd="$cmd --no-fp8"
    elif [ "$mode" == "fp8_tensorwise" ]; then
        cmd="$cmd --fp8-recipe tensorwise"
    elif [ "$mode" == "fp8_rowwise" ]; then
        cmd="$cmd --fp8-recipe rowwise"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    # Run it
    eval $cmd
    
    echo -e "${GREEN}"
    echo "✓ Completed: $log_file"
    echo -e "${NC}"
}

# Function to estimate time
estimate_time() {
    local model_size=$1
    local steps=$2
    local num_runs=$3
    
    # Rough estimates based on observed throughput (tokens/sec)
    local tokens_per_step=$((BATCH_SIZE * SEQ_LEN))
    local total_tokens=$((steps * tokens_per_step))
    
    # Throughput estimates (conservative, tokens/sec)
    local bf16_tps=4000
    local fp8_tps=6000
    
    # Adjust for model size
    case $model_size in
        "1.5B") bf16_tps=4000; fp8_tps=6000 ;;
        "3B")   bf16_tps=1400; fp8_tps=1600 ;;
        "7B")   bf16_tps=350;  fp8_tps=400  ;;
    esac
    
    # Calculate times (hours) - using shell arithmetic
    local bf16_hours=$((total_tokens / bf16_tps / 3600))
    local fp8_hours=$((total_tokens / fp8_tps / 3600))
    local avg_hours=$(( (bf16_hours + fp8_hours) / 2 ))
    local total_hours=$((avg_hours * num_runs))
    
    echo "Estimated time for $model_size @ $steps steps:"
    echo "  Total tokens: $((total_tokens / 1000000))M"
    echo "  BF16: ~${bf16_hours}h per run"
    echo "  FP8:  ~${fp8_hours}h per run"
    echo "  Total for $num_runs runs: ~${total_hours}h"
    echo ""
}

# Parse arguments
case "${1:-default}" in
    --quick)
        echo -e "${BLUE}Running quick test (100 steps each)${NC}\n"
        
        # Quick test - just verify everything works
        run_experiment "1.5B" 100 "bf16"
        run_experiment "1.5B" 100 "fp8_tensorwise"
        # run_experiment "1.5B" 100 "fp8_rowwise"
        ;;
        
    --500m)
        echo -e "${BLUE}Running 500M token experiments (1.5B model)${NC}\n"
        
        # 500M tokens = ~15000 steps at batch 32, seq 1024
        STEPS=15000
        
        estimate_time "1.5B" $STEPS 2
        
        run_experiment "1.5B" $STEPS "bf16"
        run_experiment "1.5B" $STEPS "fp8_tensorwise"
        # run_experiment "1.5B" $STEPS "fp8_rowwise"
        ;;
        
    --full)
        echo -e "${BLUE}Running full experiment suite${NC}\n"
        
        # 1.5B model - 500M tokens (~6-8h total)
        echo "=== 1.5B Model ==="
        estimate_time "1.5B" 15000 2
        run_experiment "1.5B" 15000 "bf16"
        run_experiment "1.5B" 15000 "fp8_tensorwise"
        # run_experiment "1.5B" 15000 "fp8_rowwise"
        
        # 3B model - 200M tokens (~8-10h total)
        echo "=== 3B Model ==="
        BATCH_3B=16  # Smaller batch for 3B
        estimate_time "3B" 12000 2
        run_experiment "3B" 12000 "bf16" $BATCH_3B
        run_experiment "3B" 12000 "fp8_tensorwise" $BATCH_3B
        # run_experiment "3B" 12000 "fp8_rowwise" $BATCH_3B
        ;;
        
    --1.5b-long)
        echo -e "${BLUE}Running 1.5B extended training (500M tokens, 3 modes)${NC}\n"
        
        STEPS=15000
        estimate_time "1.5B" $STEPS 2
        
        echo "Starting BF16 baseline..."
        run_experiment "1.5B" $STEPS "bf16"
        
        echo "Starting FP8 tensorwise..."
        run_experiment "1.5B" $STEPS "fp8_tensorwise"
        
        # echo "Starting FP8 rowwise..."
        # run_experiment "1.5B" $STEPS "fp8_rowwise"
        ;;
        
    --custom)
        # Custom run: ./run_experiments.sh --custom MODEL STEPS [MODES...]
        MODEL_SIZE=${2:-"1.5B"}
        STEPS=${3:-5000}
        shift 3 || true
        MODES=${@:-"bf16 fp8_tensorwise"}
        
        echo -e "${BLUE}Running custom experiment: $MODEL_SIZE, $STEPS steps${NC}\n"
        
        for mode in $MODES; do
            run_experiment "$MODEL_SIZE" "$STEPS" "$mode"
        done
        ;;
        
    --compare-recipes)
        echo -e "${BLUE}Comparing FP8 recipes (tensorwise vs rowwise)${NC}\n"
        
        # Medium run to compare FP8 recipes
        STEPS=5000
        
        run_experiment "1.5B" $STEPS "bf16"
        run_experiment "1.5B" $STEPS "fp8_tensorwise"
        # run_experiment "1.5B" $STEPS "fp8_rowwise"
        ;;
        
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --quick          Quick test (100 steps each)"
        echo "  --500m           500M token run on 1.5B model (~12h)"
        echo "  --full           Full suite: 1.5B and 3B models (~24h+)"
        echo "  --1.5b-long      1.5B model, 500M tokens, all 3 modes"
        echo "  --compare-recipes Compare tensorwise vs rowwise FP8"
        echo "  --custom MODEL STEPS [MODES...]  Custom experiment"
        echo "  --help           Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 --quick                        # Quick test"
        echo "  $0 --custom 1.5B 5000             # 1.5B, 5000 steps, bf16+fp8"
        echo "  $0 --custom 3B 3000 fp8_tensorwise # Just FP8 on 3B"
        echo ""
        echo "Output: CSV logs saved to ./experiments/"
        ;;
        
    *)
        # Default: moderate run for overnight
        echo -e "${BLUE}Running default experiments (1.5B, ~8h total)${NC}\n"
        
        STEPS=8000  # ~250M tokens, ~4h BF16 + ~3h FP8
        
        estimate_time "1.5B" $STEPS 3
        
        run_experiment "1.5B" $STEPS "bf16"
        run_experiment "1.5B" $STEPS "fp8_tensorwise"
        # run_experiment "1.5B" $STEPS "fp8_rowwise"
        ;;
esac

# Summary
echo -e "${GREEN}"
echo "============================================================"
echo "  All experiments complete!"
echo "============================================================"
echo -e "${NC}"

echo "Results saved to: $OUTPUT_DIR/"
ls -la $OUTPUT_DIR/*.csv 2>/dev/null || echo "No CSV files found"

echo ""
echo "To plot results:"
echo "  python plot_training.py ${OUTPUT_DIR}/*.csv --output plots/"
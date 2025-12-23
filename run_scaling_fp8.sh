#!/bin/bash
# run_fp8_scaling.sh
#
# FP8 Scaling Law Experiments on H100
# ====================================
#
# Three models at 20x Chinchilla optimal:
#   125M: 2.86B tokens, ~1.6-2.0 hours
#   250M: 5.31B tokens, ~4.2-5.9 hours  
#   350M: 8.35B tokens, ~11.6-13.6 hours
#
# Total estimated time: 17-22 hours
#
# Usage:
#   chmod +x run_fp8_scaling.sh
#   nohup ./run_fp8_scaling.sh > fp8_scaling.log 2>&1 &
#
# Monitor:
#   tail -f fp8_scaling.log

set -e  # Exit on error

echo "======================================================================"
echo "FP8 SCALING LAW RUNS - H100"
echo "======================================================================"
echo "Started: $(date)"
echo ""
echo "Run 1: 125M @ 2.86B tokens (~2h)"
echo "Run 2: 250M @ 5.31B tokens (~5h)"
echo "Run 3: 350M @ 8.35B tokens (~13h)"
echo "Total: ~17-22 hours"
echo "======================================================================"
echo ""

mkdir -p experiments
mkdir -p checkpoints

# ============================================
# 125M Model - 2.86B tokens (20x Chinchilla)
# ============================================
# MBS: 96, Grad Accum: 20, Effective: 1920 seqs
# Tokens/step: 1.97M, Steps: 1456

echo "[$(date)] Starting 125M FP8 run..."
echo "  Tokens: 2.86B | Steps: 1456 | Batch: 96x20"
echo ""

python train_gpt_fp8.py \
    --model-size 125M \
    --batch-size 96 \
    --grad-accum 20 \
    --steps 1456 \
    --warmup-frac 0.02 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --val-every 200 \
    --log-every 20 \
    --train-data "data/pretrain/train_*.bin" \
    --val-data "data/pretrain/val_*.bin" \
    --log-file experiments/scaling_125M_fp8.csv \
    --checkpoint-path checkpoints/scaling_125M_fp8.pt

echo ""
echo "[$(date)] 125M FP8 complete!"
echo "======================================================================"
echo ""

# ============================================
# 250M Model - 5.31B tokens (20x Chinchilla)
# ============================================
# MBS: 80, Grad Accum: 24, Effective: 1920 seqs
# Tokens/step: 1.97M, Steps: 2700

echo "[$(date)] Starting 250M FP8 run..."
echo "  Tokens: 5.31B | Steps: 2700 | Batch: 80x24"
echo ""

python train_gpt_fp8.py \
    --model-size 250M \
    --batch-size 80 \
    --grad-accum 24 \
    --steps 2700 \
    --warmup-frac 0.02 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --val-every 200 \
    --log-every 20 \
    --train-data "data/pretrain/train_*.bin" \
    --val-data "data/pretrain/val_*.bin" \
    --log-file experiments/scaling_250M_fp8.csv \
    --checkpoint-path checkpoints/scaling_250M_fp8.pt

echo ""
echo "[$(date)] 250M FP8 complete!"
echo "======================================================================"
echo ""

# ============================================
# 350M Model - 8.35B tokens (20x Chinchilla)
# ============================================
# MBS: 64, Grad Accum: 31, Effective: 1984 seqs
# Tokens/step: 2.03M, Steps: 4110

echo "[$(date)] Starting 350M FP8 run..."
echo "  Tokens: 8.35B | Steps: 4110 | Batch: 64x31"
echo ""

python train_gpt_fp8.py \
    --model-size 350M \
    --batch-size 64 \
    --grad-accum 31 \
    --steps 4110 \
    --warmup-frac 0.02 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --val-every 200 \
    --log-every 20 \
    --train-data "data/pretrain/train_*.bin" \
    --val-data "data/pretrain/val_*.bin" \
    --log-file experiments/scaling_350M_fp8.csv \
    --checkpoint-path checkpoints/scaling_350M_fp8.pt

echo ""
echo "[$(date)] 350M FP8 complete!"
echo ""

echo "======================================================================"
echo "ALL FP8 SCALING RUNS COMPLETE"
echo "======================================================================"
echo "Finished: $(date)"
echo ""
echo "Checkpoints:"
echo "  - checkpoints/scaling_125M_fp8.pt"
echo "  - checkpoints/scaling_250M_fp8.pt"
echo "  - checkpoints/scaling_350M_fp8.pt"
echo ""
echo "Logs:"
echo "  - experiments/scaling_125M_fp8.csv"
echo "  - experiments/scaling_250M_fp8.csv"
echo "  - experiments/scaling_350M_fp8.csv"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints: scp -P <port> root@<host>:checkpoints/*.pt ."
echo "  2. Plot scaling curves from CSV files"
echo "  3. Compare with BF16 baselines"
echo "======================================================================"

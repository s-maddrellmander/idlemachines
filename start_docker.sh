#!/bin/bash
# Simple Docker launcher for FP8 training
# Just starts the container and installs packages

docker run -it --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/fp8_training:/workspace \
    -v ~/.cache:/root/.cache \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.09-py3 \
    bash -c '
        pip install matplotlib seaborn pandas --quiet --break-system-packages
        echo ""
        echo "========================================="
        echo "  Ready! Packages installed."
        echo "========================================="
        echo ""
        exec bash
    '
#!/bin/bash

echo "Starting llama.cpp with CUDA support..."
echo "Using default model: gemma-3-12b-it-UD-Q4_K_XL.gguf"
echo "========================================"

docker run --gpus all -it --rm \
    -v "$(pwd):/workspace" \
    vdh/vdh-multi:cuda12.6 \
    llama-cli -m /app/models/gemma-3-12b-it-UD-Q4_K_XL.gguf \
    --ctx-size 2048 \
    --n-gpu-layers 999 \
    --prompt "$@"
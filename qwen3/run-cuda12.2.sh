#!/bin/bash

docker run -e CTX_SIZE=4096 -p 5000:5000 --gpus all -it --rm vdh/qwen3-235b-a22b-instruct-2507:cuda12.2

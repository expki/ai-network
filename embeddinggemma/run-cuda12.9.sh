#!/bin/bash

docker run -e CTX_SIZE=4096 --gpus all -it --rm vdh/embeddinggemma-300m:cuda12.9

#!/bin/bash

docker run -e CTX_SIZE=4096 -p 5000:5000 --gpus all -it --rm vdh/embeddinggemma-300m:cuda12.9

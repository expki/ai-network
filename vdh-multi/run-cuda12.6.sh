#!/bin/bash

docker run -e CTX_SIZE=4096 -p 5000:5000 --gpus all -it --rm vdh/vdh-multi:cuda12.6

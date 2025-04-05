#!/bin/bash

sudo docker run --rm -it -p 7500:7500 --gpus all vdh/nomic-embed-text-v2-moe:cuda

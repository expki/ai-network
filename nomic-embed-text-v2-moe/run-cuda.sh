#!/bin/bash

sudo docker run -d -p 7500:7500 --gpus all nomic-embed-text-v2-moe:cuda

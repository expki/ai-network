#!/bin/bash

docker run -p 5000:5000 --gpus all -it --rm vdh/jina-reranker-v1-turbo-en:cuda12.6

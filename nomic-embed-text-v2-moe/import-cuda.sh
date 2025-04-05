#/bin/bash

zstd -dc ./build/nomic-embed-text-v2-moe_cuda.tar.zst | docker load

#/bin/bash

mkdir -p ./build
sudo docker save -o ./build/nomic-embed-text-v2-moe_cuda.tar nomic-embed-text-v2-moe:cuda
zstd ./build/nomic-embed-text-v2-moe_cuda.tar -o ./build/nomic-embed-text-v2-moe_cuda.tar.zst
rm ./build/nomic-embed-text-v2-moe_cuda.tar

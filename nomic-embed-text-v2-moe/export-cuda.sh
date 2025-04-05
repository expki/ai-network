#/bin/bash

mkdir -p ./build
sudo docker save -o ./build/vdh_nomic-embed-text-v2-moe_cuda.tar vdh/nomic-embed-text-v2-moe:cuda
sudo chmod +r ./build/vdh_nomic-embed-text-v2-moe_cuda.tar
zstd ./build/vdh_nomic-embed-text-v2-moe_cuda.tar -o ./build/vdh_nomic-embed-text-v2-moe_cuda.tar.zst
sudo rm ./build/vdh_nomic-embed-text-v2-moe_cuda.tar

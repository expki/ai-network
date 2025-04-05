#/bin/bash

mkdir -p ./build
sudo docker save -o ./build/vdh_nomic-embed-text-v2-moe_rocm.tar vdh/nomic-embed-text-v2-moe:rocm
sudo chmod +r ./build/vdh_nomic-embed-text-v2-moe_rocm.tar
zstd ./build/vdh_nomic-embed-text-v2-moe_rocm.tar -o ./build/vdh_nomic-embed-text-v2-moe_rocm.tar.zst
sudo rm ./build/vdh_nomic-embed-text-v2-moe_rocm.tar

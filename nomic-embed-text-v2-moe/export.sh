#/bin/bash

mkdir -p ./build
sudo docker save -o ./build/vdh_nomic-embed-text-v2-moe.tar vdh/nomic-embed-text-v2-moe
sudo chmod +r ./build/vdh_nomic-embed-text-v2-moe.tar
zstd ./build/vdh_nomic-embed-text-v2-moe.tar -o ./build/vdh_nomic-embed-text-v2-moe.tar.zst
sudo rm ./build/vdh_nomic-embed-text-v2-moe.tar

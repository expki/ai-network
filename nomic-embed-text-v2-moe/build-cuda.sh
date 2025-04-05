#!/bin/bash

sudo docker build -t vdh/nomic-embed-text-v2-moe:cuda-temp ./cuda/
sudo docker run --gpus all -it --name vdh_nomic-embed-text-v2-moe_cuda-temp vdh/nomic-embed-text-v2-moe:cuda-temp /root/setup.sh
sudo docker commit vdh_nomic-embed-text-v2-moe_cuda-temp vdh/nomic-embed-text-v2-moe:cuda-temp 
printf 'FROM vdh/nomic-embed-text-v2-moe:cuda-temp \nCMD ["/root/environment/bin/python3", "/root/server.py"]' > ./Dockerfile
sudo docker build -t vdh/nomic-embed-text-v2-moe:cuda .
sudo docker rm vdh_nomic-embed-text-v2-moe_cuda-temp
sudo docker rmi vdh/nomic-embed-text-v2-moe:cuda-temp
rm ./Dockerfile

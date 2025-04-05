#!/bin/bash

sudo docker build -t nomic-embed-text-v2-moe:cuda-temp ./cuda/
sudo docker run --gpus all -it --name nomic-embed-text-v2-moe_cuda-temp nomic-embed-text-v2-moe:cuda-temp /root/setup.sh
sudo docker commit nomic-embed-text-v2-moe_cuda-temp nomic-embed-text-v2-moe:cuda-temp 
printf 'FROM nomic-embed-text-v2-moe:cuda-temp \nCMD ["/root/environment/bin/python3", "/root/server.py"]' > ./Dockerfile
sudo docker build -t nomic-embed-text-v2-moe:cuda .
sudo docker rm nomic-embed-text-v2-moe_cuda-temp
sudo docker rmi nomic-embed-text-v2-moe:cuda-temp
rm ./Dockerfile

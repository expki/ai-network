#!/bin/bash

sudo docker run --rm -it -p 7400:7400 --runtime=rocm vdh/nomic-embed-text-v2-moe:rocm

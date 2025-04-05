#!/bin/bash

sudo docker run --rm -it -p 7400:7400 --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri vdh/nomic-embed-text-v2-moe:rocm

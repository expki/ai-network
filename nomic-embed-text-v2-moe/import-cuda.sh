#!/bin/bash

zstd -dc ./build/vdh_nomic-embed-text-v2-moe_cuda.tar.zst | docker load

#!/bin/bash

zstd -dc ./build/vdh_nomic-embed-text-v2-moe_rocm.tar.zst | docker load

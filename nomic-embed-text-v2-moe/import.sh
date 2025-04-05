#!/bin/bash

zstd -dc ./build/vdh_nomic-embed-text-v2-moe.tar.zst | docker load

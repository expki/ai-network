#!/bin/bash

./build-cuda12.2.sh || exit 1
./build-cuda12.6.sh || exit 1
./build-cuda12.9.sh || exit 1

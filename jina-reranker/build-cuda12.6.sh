#!/bin/bash

# Default values
IMAGE_NAME="vdh/jina-reranker-v1-turbo-en"
TAG="cuda12.6"
CACHE_VOLUME="temp-$TAG-cmake"

# Build Golang
go build -o "$TAG/llama-proxy" .

# Create Docker volume for ccache if it doesn't exist
if ! docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1; then
    echo "Creating Docker volume: ${CACHE_VOLUME}"
    docker volume create "${CACHE_VOLUME}"
fi

# Enable BuildKit for Docker
export DOCKER_BUILDKIT=1

# Build the Docker image with BuildKit cache mounts
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Using ccache volume: ${CACHE_VOLUME}"
echo "BuildKit enabled for cache mount support"
docker build \
    --progress=plain \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t "${IMAGE_NAME}:${TAG}" \
    "$TAG/."

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Successfully built ${IMAGE_NAME}:${TAG}"
    echo "To run the container: docker run --gpus all -it ${IMAGE_NAME}:${TAG}"
else
    echo "Build failed"
    exit 1
fi

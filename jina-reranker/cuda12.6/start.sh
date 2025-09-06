#!/bin/bash

# Set default values for runtime environment variables
: ${THREADS:=8}
: ${THREADS_BATCH:=8}
: ${PARALLEL:=4}
: ${N_GPU_LAYERS:=9999}
: ${CACHE_TYPE_K:=q4_0}
: ${CACHE_TYPE_V:=q8_0}
# Set ctx-size default to -1 (use model default) if not specified
: ${CTX_SIZE:=-1}
: ${BATCH_SIZE:=512}

echo "Starting llama-proxy in background..."
# Run llama-proxy in background with output to stderr so we can see both services
TARGET_URL=http://localhost:8080 LISTEN_ADDR=:5000 /usr/local/bin/llama-proxy 2>&1 | sed "s/^/[llama-proxy] /" >&2 &

echo "Starting llama-server with reranker support..."
/usr/local/bin/llama-server \
  --model ${MODEL_PATH} \
  --batch-size ${BATCH_SIZE} \
  --ctx-size ${CTX_SIZE} \
  --threads ${THREADS} \
  --threads-batch ${THREADS_BATCH} \
  --parallel ${PARALLEL} \
  --n-gpu-layers ${N_GPU_LAYERS} \
  --flash-attn on \
  --host localhost \
  --port 8080 \
  --cont-batching \
  --metrics \
  --reranking \
  "$@"

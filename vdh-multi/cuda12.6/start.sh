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
TARGET_URL_CHAT=http://localhost:5001 TARGET_URL_EMBED=http://localhost:5002 TARGET_URL_RERANK=http://localhost:5003 LISTEN_ADDR=:5000 /usr/local/bin/llama-proxy 2>&1 | sed "s/^/[llama-proxy] /" >&2 &

# Start chat model server
if [ -n "${MODEL_PATH_CHAT}" ]; then
  echo "Starting llama-server for chat model on port 5001..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_CHAT} \
    --batch-size ${BATCH_SIZE} \
    --ctx-size ${CTX_SIZE} \
    --threads ${THREADS} \
    --threads-batch ${THREADS_BATCH} \
    --parallel ${PARALLEL} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --flash-attn on \
    --cache-type-k ${CACHE_TYPE_K} \
    --cache-type-v ${CACHE_TYPE_V} \
    --host localhost \
    --port 5001 \
    --cont-batching \
    --metrics \
    "$@" 2>&1 | sed "s/^/[chat-5001] /" >&2 &
fi

# Start embedding model server
if [ -n "${MODEL_PATH_EMBED}" ]; then
  echo "Starting llama-server for embedding model on port 5002..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_EMBED} \
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
    --embeddings \
    --pooling cls \
    "$@" 2>&1 | sed "s/^/[embed-5002] /" >&2 &
fi

# Start reranking model server
if [ -n "${MODEL_PATH_RERANK}" ]; then
  echo "Starting llama-server for reranking model on port 5003..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_RERANK} \
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
    "$@" 2>&1 | sed "s/^/[rerank-5003] /" >&2 &
fi

# Wait for all background processes
wait

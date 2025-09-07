#!/bin/bash

# Function to cleanup all processes
cleanup() {
    echo "Stopping all services..."
    # Kill all child processes
    pkill -P $$
    # Kill any remaining jobs
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM to cleanly stop all processes
trap cleanup INT TERM

# Set default values for runtime environment variables
: ${THREADS:=8}
: ${THREADS_BATCH:=8}
: ${N_GPU_LAYERS:=9999}
: ${CACHE_TYPE_K:=q4_0}
: ${CACHE_TYPE_V:=q4_0}
: ${CTX_SIZE_CHAT:=2048}
: ${CTX_SIZE_EMBED:=2048}
: ${CTX_SIZE_RERANK:=2048}
: ${BATCH_SIZE_CHAT:=2048}
: ${BATCH_SIZE_EMBED:=2048}
: ${BATCH_SIZE_RERANK:=2048}
: ${PARALLEL_CHAT:=10}
: ${PARALLEL_EMBED:=10}
: ${PARALLEL_RERANK:=10}

# Array to store PIDs
declare -a PIDS=()

echo "Starting llama-proxy in background..."
# Run llama-proxy in background with output to stderr so we can see both services
TARGET_URL_CHAT=http://localhost:5001 TARGET_URL_EMBED=http://localhost:5002 TARGET_URL_RERANK=http://localhost:5003 LISTEN_ADDR=:5000 /usr/local/bin/llama-proxy 2>&1 | sed "s/^/[llama-proxy] /" >&2 &
PIDS+=($!)

# Start chat model server
if [ -n "${MODEL_PATH_CHAT}" ]; then
  echo "Starting llama-server for chat model on port 5001..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_CHAT} \
    --batch-size ${BATCH_SIZE_CHAT} \
    --ubatch-size ${BATCH_SIZE_CHAT} \
    --ctx-size $((CTX_SIZE_CHAT * PARALLEL_CHAT)) \
    --threads ${THREADS} \
    --threads-batch ${THREADS_BATCH} \
    --parallel ${PARALLEL_CHAT} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --flash-attn on \
    --cache-type-k ${CACHE_TYPE_K} \
    --cache-type-v ${CACHE_TYPE_V} \
    --host localhost \
    --port 5001 \
    --cont-batching \
    --metrics \
    "$@" 2>&1 | sed "s/^/[chat-5001] /" >&2 &
  PIDS+=($!)
fi

# Start embedding model server
if [ -n "${MODEL_PATH_EMBED}" ]; then
  echo "Starting llama-server for embedding model on port 5002..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_EMBED} \
    --batch-size ${BATCH_SIZE_EMBED} \
    --ubatch-size ${BATCH_SIZE_EMBED} \
    --ctx-size $((CTX_SIZE_EMBED * PARALLEL_EMBED)) \
    --threads ${THREADS} \
    --threads-batch ${THREADS_BATCH} \
    --parallel ${PARALLEL_EMBED} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --flash-attn on \
    --host localhost \
    --port 5002 \
    --cont-batching \
    --metrics \
    --embeddings \
    --pooling cls \
    "$@" 2>&1 | sed "s/^/[embed-5002] /" >&2 &
  PIDS+=($!)
fi

# Start reranking model server
if [ -n "${MODEL_PATH_RERANK}" ]; then
  echo "Starting llama-server for reranking model on port 5003..."
  /usr/local/bin/llama-server \
    --model ${MODEL_PATH_RERANK} \
    --batch-size ${BATCH_SIZE_RERANK} \
    --ubatch-size ${BATCH_SIZE_RERANK} \
    --ctx-size $((CTX_SIZE_RERANK * PARALLEL_RERANK)) \
    --threads ${THREADS} \
    --threads-batch ${THREADS_BATCH} \
    --parallel ${PARALLEL_RERANK} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --flash-attn on \
    --host localhost \
    --port 5003 \
    --cont-batching \
    --metrics \
    --reranking \
    "$@" 2>&1 | sed "s/^/[rerank-5003] /" >&2 &
  PIDS+=($!)
fi

# Monitor all processes - if any dies, kill all and exit
echo "Monitoring processes: ${PIDS[@]}"
while true; do
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Process $pid terminated. Stopping all services..."
            cleanup
        fi
    done
    sleep 1
done

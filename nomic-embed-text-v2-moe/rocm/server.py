#!/root/environment/bin/python3

### Powered by nomic-embed-text-v2-moe (https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
## @misc{nussbaum2025trainingsparsemixtureexperts,
##       title={Training Sparse Mixture Of Experts Text Embedding Models}, 
##       author={Zach Nussbaum and Brandon Duderstadt},
##       year={2025},
##       eprint={2502.07972},
##       archivePrefix={arXiv},
##       primaryClass={cs.CL},
##       url={https://arxiv.org/abs/2502.07972}, 
## }

citation = """
Powered by nomic-embed-text-v2-moe (https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
@misc{nussbaum2025trainingsparsemixtureexperts,
    title={Training Sparse Mixture Of Experts Text Embedding Models}, 
    author={Zach Nussbaum and Brandon Duderstadt},
    year={2025},
    eprint={2502.07972},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.07972}, 
}
"""

import os
import asyncio
import logging
import zstandard as zstd
import numpy as np
import threading
import json

from platform import processor
from psutil import cpu_count
from torch.cuda import is_available, device_count, get_device_properties

from quart import Quart, request, jsonify, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config


import ai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

# Initialize the Quart app
app = Quart(__name__)

# Shutdown cleanup function.
@app.after_serving
async def shutdown():
    logger.error("Shutdown request received")
    await ai.shutdown()

# Middleware to decompress request data if Content-Encoding is zstd.
@app.before_request
async def decompress_request():
    data = None
    json_data = None

    # Read
    try:
        data = await request.get_data()
        if data == b'':
            data = None
    except:
        data = None

    # Decompress
    if request.headers.get("Content-Encoding", "").strip().lower() == "zstd":
        try:
            decompressor = zstd.ZstdDecompressor()
            data = decompressor.decompress(data)
        except Exception as e:
            return jsonify({"error": f"failed to decompress request body: {e}"}), 400
    
    # JSON
    if data != None:
        try:
            json_data = json.loads(data)
        except json.JSONDecodeError:
            json_data = None

    # Store
    request._cached_data = data
    request._cached_json = json_data

# Middleware to compress response data if the client accepts zstd encoding.
@app.after_request
async def compress_response(response):
    accept_encoding = request.headers.get("Accept-Encoding", "").strip().lower()
    # Only compress if the client supports zstd and if the response has a body.
    if "zstd" in accept_encoding and response.status_code == 200:
        # Retrieve response data
        data = await response.get_data()
        try:
            compressor = zstd.ZstdCompressor()
            compressed = compressor.compress(data)
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "zstd"
            response.headers["Content-Length"] = str(len(compressed))
        except Exception as e:
            # If compression fails, log the error or take alternative action.
            app.logger.error(f"Response compression failed: {e}")
    return response

@app.route('/', methods=['GET'])
async def root_request():
    # Always respond
    return Response(
        "nomic-embed-text-v2-moe is running",
        content_type='text/plain',
        status=200
    )

@app.route('/status', methods=['GET'])
async def status_request():
    # Check if the processor is currently busy
    is_busy = await ai.is_processing()
    
    # Respond status
    return Response(
        f"nomic-embed-text-v2-moe is {'busy' if is_busy else 'ready'}",
        content_type='text/plain',
        status=(102 if is_busy else 200)
    )

@app.route('/id', methods=['GET'])
async def id_request():
    try:
        items = [{"name": processor(), "id": "cpu", "cores": str(cpu_count(logical=True)), "threads": str(cpu_count(logical=True))}]
        for i in range(device_count()):
            props = get_device_properties(i)
            items.append({"name": props.name, "id": props.uuid})
        
        # Respond id
        return jsonify({
            "devices": items
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving device id: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e)}}), 500
    
@app.route('/total', methods=['GET'])
async def total_request():
    try:        
        # Respond total
        return jsonify({
            "total": await ai.total()
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving total: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e)}}), 500

@app.route('/api/embedding', methods=['GET', 'POST']) # OpenAPI compatible
async def process_openapi_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return jsonify({"error": {"message": "Missing request body"}}), 400

        # Extract text list
        if 'prompt' not in data:
            return jsonify({"error": {"message": "Missing 'prompt' field"}}), 400

        textList = data['prompt']
        if not isinstance(textList, list):
            textList = [textList]
        
        # Process request
        try:
            # Wait for the response
            result = await ai.process_request(textList)
            if result is None:
                return jsonify({"error": {"message": "AI returned empty embedding"}}), 500
            
            # Return response to client
            return jsonify({
                    "model": "nomic-embed-text-v2-moe",
                    "data": [{"embedding": embeddingArray, "index": i} for i, embeddingArray in enumerate(result.tolist())]
                }), 200

        except asyncio.CancelledError:
            return jsonify({"error": {"message": "Client cancelled the OpenAPI task"}}), 499
        
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing OpenAPI request: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e)}}), 400

@app.route('/api/embed', methods=['GET', 'POST']) # Ollama compatible
async def process_ollama_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return jsonify({"error": {"message": "Missing request body"}}), 400

        # Extract text list
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        textList = data['input']
        if not isinstance(textList, list):
            textList = [textList]
        
        try:
            # Wait for the response
            result = await ai.process_request(textList)
            if result is None:
                return jsonify({"error": {"message": "AI returned empty embedding"}}), 500
            
            # Return response to client
            return jsonify({
                    "model": "nomic-embed-text-v2-moe",
                    "embeddings": result.tolist(),
                }), 200

        except asyncio.CancelledError:
            return jsonify({"error": {"message": "Client cancelled the Ollama task"}}), 499
        
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing Ollama request: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e)}}), 500

@app.route('/vdh/embed', methods=['GET', 'POST']) # VDH compatible
async def process_vdh_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return jsonify({"error": "Missing request body"}), 400

        # Extract text list
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        textList = data['input']
        if not isinstance(textList, list):
            textList = [textList]
        
        # Process request
        try:
            # Wait for the response
            result = await ai.process_request(textList)
            if result is None:
                return jsonify({"error": "AI returned empty embedding"}), 400
            
            # Quantize response q8_0(-1,1)
            embeddingMatrix = np.multiply(np.divide(np.subtract(result, -1), 2), 255).astype(np.uint8).tobytes(order='C')
            # Return response to client
            return Response(
                embeddingMatrix,
                status=200,
                content_type='application/octet-stream',
            )
        except asyncio.CancelledError:
            return jsonify({"error": "Client cancelled the VDH task"}), 499
        
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing VDH request: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e)}}), 500

if __name__ == '__main__':
    print(citation)
    if not is_available():
        raise Exception("ROCm is not available")
    
    # Start the worker thread.
    worker_thread = threading.Thread(target=ai.text_processor, daemon=True)
    worker_thread.start()

    config = Config()
    # Bind the server to a host and port.
    config.bind = ["0.0.0.0:7500"]
    # Connection settings
    config.max_concurrency = 1000
    config.backlog = 2048
    config.keep_alive_timeout = 45
    config.graceful_timeout = 5
    # Enable HTTP/2 support (ALPN protocols).
    config.alpn_protocols = ["h2", "http/1.1"]
    # Enable TLS
    config.certfile = "/etc/ssl/certs/server.crt"
    config.keyfile = "/etc/ssl/private/server.key"

    # Run the Hypercorn server with HTTP/2 support.
    asyncio.run(serve(app, config))

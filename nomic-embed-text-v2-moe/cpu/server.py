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

import json
import asyncio
import logging
from hypercorn.config import Config
from hypercorn.asyncio import serve
from quart import Quart, Response, request
import zstandard as zstd
import numpy as np
import signal

import ai

class ZstdCompressionMiddleware:
    def __init__(self, app, minimum_size: int = 1000):
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope, receive, send):
        # Process the request and generate a response using the app
        response = await self.app(scope, receive, send)

        # Check if the client supports zstd compression
        accept_encoding = dict(scope.get('headers', {})).get(b'accept-encoding', b'').decode('utf-8')

        # Only compress with zstd if the client supports it and the response is large enough
        if response is not None and 'zstd' in accept_encoding and len(response.body) > self.minimum_size:
            compressor = zstd.ZstdCompressor()
            compressed_body = compressor.compress(response.body)

            # Modify response to use compressed body
            response = Response(
                compressed_body,
                status=response.status_code,
                headers=response.headers
            )
            response.headers['Content-Encoding'] = 'zstd'

        # Send the final response
        await send(response)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

# Initialize the Quart app
app = Quart(__name__)

# Apply ZstdCompressionMiddleware to the Quart app
app.asgi_app = ZstdCompressionMiddleware(app.asgi_app)

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

@app.route('/api/embedding', methods=['GET', 'POST']) # OpenAPI compatible
async def process_openapi_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return Response(
                json.dumps({"error": {"message": "Missing request body"}}), 
                status=400, 
                content_type='application/json'
            )

        # Extract text list
        if 'prompt' not in data:
            return Response(
                json.dumps({"error": {"message": "Missing 'prompt' field"}}), 
                status=400, 
                content_type='application/json'
            )
        textList = data['prompt']
        if not isinstance(textList, list):
            textList = [textList]
        
        # Process request
        task = asyncio.create_task(ai.process_request(textList))
        try:
            # Wait for the response
            result = await task
            if result is None:
                return Response(
                    "AI returned empty embedding",
                    status=500
                )
            # Return response to client
            return Response(
                json.dumps({
                    "model": "nomic-embed-text-v2-moe",
                    "data": [{"embedding": embeddingArray, "index": i} for i, embeddingArray in enumerate(result.tolist())]
                }),
                status=200,
                content_type='application/json',
            )
        except asyncio.CancelledError:
            # Handle client canceled request
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return Response(
                "Client cancelled the OpenAPI task",
                status=499
            )
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing OpenAPI request: {e}", exc_info=True)
        return Response(
            json.dumps({"error": {"message": str(e)}}), 
            status=500, 
            content_type='application/json'
        )

@app.route('/api/embed', methods=['GET', 'POST']) # Ollama compatible
async def process_ollama_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return Response(
                json.dumps({"error": {"message": "Missing request body"}}), 
                status=400, 
                content_type='application/json'
            )

        # Extract text list
        if 'input' not in data:
            return Response(
                json.dumps({"error": "Missing 'input' field"}), 
                status=400, 
                content_type='application/json'
            )
        textList = data['input']
        if not isinstance(textList, list):
            textList = [textList]
        
        # Process request
        task = asyncio.create_task(ai.process_request(textList))
        try:
            # Wait for the response
            result = await task
            if result is None:
                return Response(
                    "AI returned empty embedding",
                    status=500
                )
            # Return response to client
            return Response(
                json.dumps({
                    "model": "nomic-embed-text-v2-moe",
                    "embeddings": result.tolist(),
                }),
                status=200,
                content_type='application/json',
            )
        except asyncio.CancelledError:
            # Handle client canceled request
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return Response(
                "Client cancelled the Ollama task",
                status=499
            )
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing Ollama request: {e}", exc_info=True)
        return Response(
            json.dumps({"error": {"message": str(e)}}), 
            status=500, 
            content_type='application/json'
        )

@app.route('/vdh/embed', methods=['GET', 'POST']) # VDH compatible
async def process_vdh_request():
    try:
        # Parse the JSON request
        data = await request.get_json()
        if data is None:
            return Response(
                json.dumps({"error": {"message": "Missing request body"}}), 
                status=400, 
                content_type='application/json'
            )

        # Extract text list
        if 'input' not in data:
            return Response(
                json.dumps({"error": "Missing 'input' field"}), 
                status=400, 
                content_type='application/json'
            )
        textList = data['input']
        if not isinstance(textList, list):
            textList = [textList]
        
        # Process request
        task = asyncio.create_task(ai.process_request(textList))
        try:
            # Wait for the response
            result = await task
            if result is None:
                return Response(
                    "AI returned empty embedding",
                    status=500
                )
            # Quantize response q8_0(-1,1)
            embeddingMatrix = np.multiply(np.divide(np.subtract(result, -1), 2), 255).astype(np.uint8).tobytes(order='C')
            # Return response to client
            return Response(
                embeddingMatrix,
                status=200,
                content_type='application/octet-stream',
            )
        except asyncio.CancelledError:
            # Handle client canceled request
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return Response(
                "Client cancelled the VDH task",
                status=499
            )
    except Exception as e:
        # Handle uncaught exception
        logger.error(f"Error processing VDH request: {e}", exc_info=True)
        return Response(
            json.dumps({"error": {"message": str(e)}}), 
            status=500, 
            content_type='application/json'
        )

async def serve_app(app, config):
    print("Hypercorn server is starting")
    try:
        # Running the server, this will keep it alive until shutdown
        await serve(app, config)
    except Exception:
        pass  # Gracefully handle the server shutdown here
    finally:
        await ai.shutdown()
    print("Hypercorn server has stopped")

async def main():
    print(citation)
    config = Config()
    config.bind = ["0.0.0.0:7300"]
    config.h2 = True
    config.cors_allowed_origins = "*"
    config.certfile = "/etc/ssl/certs/server.crt"
    config.keyfile = "/etc/ssl/private/server.key"

    # Scheduling both functions to run concurrently
    task1 = asyncio.create_task(ai.text_processor())
    task2 = asyncio.create_task(serve_app(app, config))

    # Wait for both tasks to finish
    await asyncio.gather(task1, task2)

async def cancel_tasks(loop, msg):
    print(msg)
    # Cancel all tasks running in the event loop
    for task in asyncio.all_tasks(loop):
        task.cancel()
    await asyncio.sleep(1)
    loop.stop()

if __name__ == '__main__':
    # Register the signal handler for Ctrl+C (SIGINT) to cancel the running tasks
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(cancel_tasks(loop, "SIGINT - Ctrl+C")))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(cancel_tasks(loop, "SIGTERM - Termination request")))
    loop.add_signal_handler(signal.SIGHUP, lambda: asyncio.create_task(cancel_tasks(loop, "SIGHUP - Reload or restart signal")))

    try:
        # Run the main task in the event loop
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Program interrupted by user.")

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
import threading
import queue
import asyncio
import time
import uuid
import logging
from hypercorn.config import Config
from hypercorn.asyncio import serve
from quart import Quart, Response, request
import zstandard as zstd
import numpy as np

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('http2_server')

# Create a queue for text processing with max size of 10
text_queue = queue.Queue(maxsize=10)
# Create a dictionary to store results
results = {}
# Create a lock for the results dictionary
results_lock = threading.Lock()
# Create a condition variable for signaling when results are ready
result_ready = threading.Condition()
# Create a condition variable for signaling when queue space is available
queue_space_available = threading.Condition()
# Flag to indicate if the processor is busy
processor_busy = False
processor_lock = threading.Lock()

# Initialize the Quart app
app = Quart(__name__)

# Apply ZstdCompressionMiddleware to the Quart app
app.asgi_app = ZstdCompressionMiddleware(app.asgi_app)

# Worker function that processes text in the queue
def text_processor():
    global processor_busy

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    # Use CUDA
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")
    device = torch.device("cuda")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('/root/model')
    model = AutoModel.from_pretrained('/root/model', trust_remote_code=True).to(device)

    logger.info(f"Server is ready")

    # Only use half precision if the device supports it
    try:
        model = model.half()
        dataType = torch.float16
    except Exception as e:
        print("Warning: model.half() failed:", e)
        dataType = torch.float32

    # Enable FlashAttention if supported
    if hasattr(model, "enable_flash_attention"):
        try:
            model.enable_flash_attention()
        except Exception as e:
            print("Warning: enable_flash_attention failed:", e)
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    # Disable model training
    model.eval()
    
    while True:
        try:
            # Get request_id and text from the queue
            request_id, sentences = text_queue.get()
            
            # Mark the processor as busy
            with processor_lock:
                processor_busy = True
            
            logger.info(f"Processing request {request_id}, queue size: {text_queue.qsize()}/{text_queue.maxsize}")

            try:
                # Tokenize and move to device
                encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

                # Autocast based on actual device type
                with torch.amp.autocast(device_type=device.type, dtype=dataType):
                    with torch.inference_mode():
                        model_output = model(**encoded_input)

                # Mean pooling and normalize
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            except Exception as e:
                logger.error(f"Processing request {request_id} failed: {e}")
                embeddings = None
            finally:
                # Store the result and notify
                with results_lock:
                    results[request_id] = embeddings
                    # Notify all waiting threads that a result is ready
                    with result_ready:
                        result_ready.notify_all()
                
                # Mark the processor as not busy
                with processor_lock:
                    processor_busy = False
                
                # Mark the task as done
                text_queue.task_done()
                
                # Notify that space is available in the queue
                with queue_space_available:
                    queue_space_available.notify_all()
            
        except Exception as e:
            logger.error(f"Error in text processor: {e}")
            # Make sure to reset the busy flag in case of error
            with processor_lock:
                processor_busy = False
                
            # Notify that space is available in the queue in case of error
            with queue_space_available:
                queue_space_available.notify_all()

# Start the text processor thread
processor_thread = threading.Thread(target=text_processor, daemon=True)
processor_thread.start()

async def process_request(request, textList):
    # Generate a unique request ID using UUID
    request_id = str(uuid.uuid4())
    
    # Create a shared flag to indicate client disconnection
    client_disconnected = {'value': False}
    
    # Check if the queue is full and wait for space if needed
    # We'll do this in a background thread to avoid blocking the event loop
    if text_queue.full():
        logger.info(f"Queue is full, request {request_id} waiting for space")
        
        # Wait for space in a separate thread
        def wait_for_space_thread():
            while True:
                # Check if client has disconnected
                if client_disconnected['value']:
                    logger.info(f"Client disconnected while waiting for queue space: {request_id}")
                    return False
                
                # Try to put in queue without blocking
                try:
                    # If we can put it in the queue without waiting, do so
                    text_queue.put_nowait((request_id, textList))
                    return True
                except queue.Full:
                    # Queue is still full, wait for notification
                    with queue_space_available:
                        # Wait with timeout to periodically check for disconnection
                        queue_space_available.wait(0.5)
        
        # Run the waiting function in a thread pool
        loop = asyncio.get_event_loop()
        added_to_queue = await loop.run_in_executor(None, wait_for_space_thread)
        
        if not added_to_queue:
            # Client disconnected while waiting, don't process
            logger.info(f"Client disconnected while waiting for queue space: {request_id}")
            return Response("", status=499)  # Client Closed Request
    else:
        # Queue has space, add immediately
        text_queue.put((request_id, textList))
        logger.info(f"Added request {request_id} to queue, size now: {text_queue.qsize()}/{text_queue.maxsize}")
    
    # Set up a task to monitor client disconnection
    async def monitor_client_connection():
        try:
            logger.info(f"Starting connection monitor for request {request_id}")
            
            # In Quart/ASGI, we can detect disconnection by checking the connection state
            # This approach uses a simple timeout-based check
            start_time = time.time()
            
            while True:
                # Short sleep to check periodically
                await asyncio.sleep(0.1)
                
                try:
                    # Try to access request data - this will fail if client disconnected
                    await request.get_data(parse=False, cache=True)
                except asyncio.CancelledError:
                    logger.info(f"Request {request_id} cancelled")
                    break
                except Exception as e:
                    logger.info(f"Client disconnected (detected via exception): {e}")
                    break
                
                # Check if we've been running too long (backup timeout)
                if time.time() - start_time > 90:  # 90 second max
                    logger.info(f"Monitor timeout for request {request_id}")
                    break
                    
                # If the result is already available, we can stop monitoring
                with results_lock:
                    if request_id in results:
                        logger.info(f"Result ready for {request_id}, stopping monitor")
                        break
        except Exception as e:
            logger.error(f"Connection monitoring error: {e}")
        finally:
            logger.info(f"Connection monitor ending for {request_id}, marking as disconnected")
            # Mark as disconnected
            client_disconnected['value'] = True
            # When client disconnects, notify all waiting threads
            with result_ready:
                result_ready.notify_all()
    
    # Start monitoring client connection in the background
    disconnect_task = asyncio.create_task(monitor_client_connection())
    
    # Wait for the result using a background thread to avoid blocking the event loop
    def wait_for_result():
        while True:
            # Check if the client has disconnected
            if client_disconnected['value']:
                return None
            
            # Check if the result is available
            with results_lock:
                if request_id in results:
                    return results.pop(request_id)
            
            # Wait for notification that a result is ready
            with result_ready:
                # Wait with a short timeout to allow for periodic checking
                result_ready.wait(0.1)
    
    # Run the waiting function in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, wait_for_result)
    
    # Cancel the disconnect monitoring task if it's still running
    if not disconnect_task.done():
        disconnect_task.cancel()
    
    if result is None:
        # Clean up any pending results for this request
        with results_lock:
            if request_id in results:
                result = results.pop(request_id)
                logger.info(f"Found result for {request_id} during cleanup: {result[:30]}...")
        
        # If client disconnected, we don't need to send a response
        if client_disconnected['value']:
            return None
        else:
            logger.info(f"Request {request_id} timed out without client disconnect")
            raise Exception(f"Processing took too long: {request_id}")
    
    # Return the result
    return result

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
    with processor_lock:
        is_busy = processor_busy
    
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
        embeddingMatrix = await process_request(request, textList)
        if embeddingMatrix is None:
            return Response(
                "",
                status=499 # Client Closed Request
            )
        embeddingMatrix = embeddingMatrix.tolist()

        # Return response
        return Response(
            json.dumps({
                "model": "nomic-embed-text-v2-moe",
                "data": [{"embedding": embeddingArray, "index": i} for i, embeddingArray in enumerate(embeddingMatrix)]
            }),
            status=200,
            content_type='application/json',
        )

    except Exception as e:
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
                json.dumps({"error": "Missing request body"}), 
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
        embeddingMatrix = await process_request(request, textList)
        if embeddingMatrix is None:
            return Response(
                "",
                status=499 # Client Closed Request
            )
        embeddingMatrix = embeddingMatrix.tolist()

        # Return response
        return Response(
            json.dumps({
                "model": "nomic-embed-text-v2-moe",
                "embeddings": embeddingMatrix,
            }),
            status=200,
            content_type='application/json',
        )

    except Exception as e:
        logger.error(f"Error processing ollama request: {e}", exc_info=True)
        return Response(
            json.dumps({"error": str(e)}), 
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
        embeddingMatrix = await process_request(request, textList)
        if embeddingMatrix is None:
            return Response(
                "",
                status=499 # Client Closed Request
            )
        embeddingMatrix = np.multiply(np.divide(np.subtract(embeddingMatrix, -1), 2), 255).astype(np.uint8).tobytes(order='C')

        # Return response
        return Response(
            embeddingMatrix,
            status=200,
            content_type='application/octet-stream',
        )

    except Exception as e:
        logger.error(f"Error processing OpenAPI request: {e}", exc_info=True)
        return Response(
            json.dumps({"error": {"message": str(e)}}), 
            status=500, 
            content_type='application/json'
        )

if __name__ == '__main__':
    print(citation)
    config = Config()
    config.bind = ["0.0.0.0:7500"]
    config.h2 = True
    config.cors_allowed_origins = "*"
    config.certfile = "/etc/ssl/certs/server.crt"
    config.keyfile = "/etc/ssl/private/server.key"
    
    asyncio.run(serve(app, config))

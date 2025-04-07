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

import asyncio
import uuid
import logging
import warnings
import os
from queue import Queue, Full

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger('processor')

# Shared objects between threads.
text_queue = Queue(maxsize=2)
processor_busy = False

# Worker function that processes text in the queue
def text_processor():
    logger.info("Starting text processor")
    global processor_busy

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    # Low memory
    lowMemory = int(os.getenv("LOW_MEMORY", "0").strip().lower())
    logger.info(f"Low Memory Mode is {"DISABLED" if lowMemory <= 0 else f"ENABLED={lowMemory}"}")

    # Load tokenizer and model
    logger.info("Loading models")
    tokenizer = AutoTokenizer.from_pretrained('/root/model')
    model = AutoModel.from_pretrained('/root/model', trust_remote_code=True)

    # Only use half precision if the device supports it
    logger.info("Switch to half precision")
    try:
        model = model.half()
        dataType = torch.float16
    except Exception as e:
        print("Warning: model.half() failed:", e)
        dataType = torch.float32
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    # Disable model training
    logger.info("Preparing model")
    model.eval()

    logger.info(f"Server is ready")
    while True:
        try:
            # Get request_id and text from the queue
            request_id, textList, response_queue = text_queue.get()
            if request_id is None:
                logger.info("text processor has stopped")
                return # shutdown
            logger.info(f"{request_id}: received from queue")
            
            # Mark the processor as busy
            processor_busy = True
            
            try:
                # Autocast based on actual device type
                logger.info(f"{request_id}: AI producing embedding")
                with torch.inference_mode(), torch.amp.autocast(device_type='cpu', dtype=dataType):
                    if lowMemory >= 1:
                        embeddings_list = []
                        for i in range(0, len(textList), lowMemory):  # Process `lowMemory` items at a time
                            # Get the current batch of `lowMemory` items
                            batch_texts = textList[i:i + lowMemory]
                            batch_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                            
                            # Forward pass
                            embedding = model(**batch_input)
                            embedding = mean_pooling(embedding, batch_input['attention_mask'])
                            embedding = F.normalize(embedding, p=2, dim=1)
                            embeddings_list.append(embedding)
                            del batch_input
                        
                        # Combine all embeddings and convert to numpy
                        embeddings = torch.cat(embeddings_list, dim=0).numpy()
                    else:
                        # Process all items at once (normal memory mode)
                        encoded_input = tokenizer(textList, padding=True, truncation=True, return_tensors='pt')
                        
                        # Forward pass
                        embeddings = model(**encoded_input)
                        embeddings = mean_pooling(embeddings, encoded_input['attention_mask'])
                        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
                        del encoded_input
            except Exception as e:
                logger.error(f"{request_id}: request failed: {str(e)}")
                embeddings = None
            finally:
                # Store the result and notify
                logger.info(f"{request_id}: returning result")
                response_queue.put(embeddings)
                text_queue.task_done()
        except Exception as e:
            logger.error(f"Error in text processor: {str(e)}")
        finally:
            processor_busy = False

async def process_request(textList):
    loop = asyncio.get_running_loop()
    
    # Generate a unique request ID using UUID
    request_id = str(uuid.uuid4())

    # Create response queue
    response_queue = Queue(maxsize=1)

    # Write to the queue
    logger.info(f"{request_id}: queueing request")
    try:
        text_queue.put_nowait((request_id, textList, response_queue))
    except Full:
        logger.info(f"{request_id}: queue is full, waiting")
        await loop.run_in_executor(None, text_queue.put, (request_id, textList, response_queue))

    # Since response_queue.get() is blocking and not awaitable,
    # run it in an executor so it doesn't block the event loop.
    result = await loop.run_in_executor(None, response_queue.get)
    logger.info(f"{request_id}: received result")

    # Return result
    return result

async def is_processing() -> bool:
    return processor_busy

async def shutdown():
    text_queue.put((None, None, None))

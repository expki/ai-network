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

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger('processor')

# Create a queue for text processing with max size of 1
text_queue = asyncio.Queue(maxsize=1)

# Create a dictionary to store results
results_lock = asyncio.Lock()
result_ready = asyncio.Condition()
results = {}

# Flag to indicate if the processor is busy
processor_lock = asyncio.Lock()
processor_busy = False

# Worker function that processes text in the queue
async def text_processor():
    logger.info("Starting text processor")
    global processor_busy

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    lowMemory = False
    envLowMemory = os.getenv("LOW_MEMORY", "0").strip().lower()
    if envLowMemory == "1" or envLowMemory == "true":
        lowMemory = True
    logger.info(f"Low Memory Mode is {"ENABLED" if lowMemory else "DISABLED"}")

    # Use CUDA
    device = torch.device("cuda")
    logger.info(f"Using GPU device: {torch.cuda.get_device_name(device)}")

    # Load tokenizer and model
    logger.info("Loading models")
    tokenizer = AutoTokenizer.from_pretrained('/root/model')
    model = AutoModel.from_pretrained('/root/model', trust_remote_code=True).to(device)

    # Only use half precision if the device supports it
    logger.info("Switch to half precision")
    try:
        model = model.half()
        dataType = torch.bfloat16
    except Exception as e:
        print("Warning: model.half() failed:", e)
        dataType = torch.float32

    # Enable FlashAttention if supported
    if hasattr(model, "enable_flash_attention"):
        logger.info("Enabling FlashAttention")
        try:
            model.enable_flash_attention()
        except Exception as e:
            print("Warning: enable_flash_attention failed:", e)
    
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
            request_id, textList = await text_queue.get()
            if request_id is None and textList is None:
                logger.info("text processor has stopped")
                return # shutdown
            logger.info(f"{request_id}: received from queue")
            
            # Mark the processor as busy
            async with processor_lock:
                processor_busy = True
            
            try:
                # Autocast based on actual device type
                logger.info(f"{request_id}: AI producing embedding")
                with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=dataType):
                    if lowMemory:
                        embeddings_list = []
                        for text in textList:
                            single_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                            single_input = {k: v.to(device) for k, v in single_input.items()}
                            embedding = model(**single_input)
                            embedding = mean_pooling(embedding, single_input['attention_mask'])
                            embedding = F.normalize(embedding, p=2, dim=1)
                            embeddings_list.append(embedding.cpu())
                        embeddings = torch.cat(embeddings_list, dim=0).numpy()
                        torch.cuda.empty_cache()
                    else:
                        encoded_input = tokenizer(textList, padding=True, truncation=True, return_tensors='pt')
                        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                        embeddings = model(**encoded_input)
                        embeddings = mean_pooling(embeddings, encoded_input['attention_mask'])
                        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            except Exception as e:
                logger.error(f"{request_id}: request failed: {str(e)}")
                embeddings = None
            finally:
                # Store the result and notify
                logger.info(f"{request_id}: storing result")
                async with results_lock:
                    results[request_id] = embeddings
                
                # Mark the task as done
                logger.info(f"{request_id}: notifying request processed")
                text_queue.task_done()

                # Notify everyone of a result that is ready
                logger.info(f"{request_id}: notifying of result ready")
                async with result_ready:
                    result_ready.notify_all()
        except Exception as e:
            logger.error(f"Error in text processor: {str(e)}")
        finally:
            async with processor_lock:
                processor_busy = False

async def process_request(textList):
    # Generate a unique request ID using UUID
    request_id = str(uuid.uuid4())

    # Write to the queue
    logger.info(f"{request_id}: adding to queue")
    await text_queue.put((request_id, textList))

    # Wait for the result
    logger.info(f"{request_id}: waiting for result")
    while True:
        # check if result is ready
        async with results_lock:
            if request_id in results:
                logger.info(f"{request_id}: received result")
                return results.pop(request_id)
        # wait for notification of result ready
        async with result_ready:
            await result_ready.wait()

async def is_processing() -> bool:
    async with processor_lock:
        is_busy = processor_busy
    return is_busy

async def shutdown():
    await text_queue.put((None, None))

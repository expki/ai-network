#!/root/environment/bin/python3

from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# Save to local directory
tokenizer.save_pretrained('/root/model')
model.save_pretrained('/root/model')

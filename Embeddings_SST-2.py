from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaModel
import torch

# Load the SST-2 dataset
dataset = load_dataset('glue', 'sst2')

# Load the LLaMA tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('meta/llama-3.1')
model = LlamaModel.from_pretrained('meta/llama-3.1')

# Tokenize function
def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Function to extract embeddings
def embed_sentences(batch):
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'].squeeze(1))
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return {'embeddings': embeddings}

# Embed the tokenized dataset
embedded_dataset = tokenized_dataset.map(embed_sentences, batched=True)

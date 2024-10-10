from transformers import LlamaTokenizer, LlamaModel

# Load the LLaMA tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('meta/llama-3.1')
model = LlamaModel.from_pretrained('meta/llama-3.1')

# Tokenize your text data
text = "Example sentence for generating latent space representation."
inputs = tokenizer(text, return_tensors='pt')

# Generate embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

print("Latent Space Representation:", embeddings)

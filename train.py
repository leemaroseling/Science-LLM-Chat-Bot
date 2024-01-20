import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pickle

# Load CSV dataset
df = pd.read_csv('data.csv')
all_text = ' '.join(df['text'].astype(str).tolist())

# Tokenize the text
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_text = tokenizer.encode(all_text, return_tensors="pt")

# Truncate the tokenized text to the maximum sequence length
max_sequence_length = 1024
tokenized_text = tokenized_text[:, :max_sequence_length]

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the device to CPU
device = torch.device("cpu") #cuda
model.to(device)
tokenized_text = tokenized_text.to(device)

# Fine-tune the model on your dataset
model.train()
model.resize_token_embeddings(len(tokenizer))

# Train for a few epochs (adjust as needed)
for epoch in range(3):
    # Forward pass
    outputs = model(tokenized_text, labels=tokenized_text)

    # Compute loss
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Update model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
# model.save_pretrained("your_saved_model")
with open("your_model.pkl", "wb") as f:
    pickle.dump((model, tokenizer), f)
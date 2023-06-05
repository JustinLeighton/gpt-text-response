# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:18:50 2023

@author: Justin Leighton
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# Load the pre-trained model and tokenizer
model_name = 'gpt2'  # or any other model you want to modify
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a new padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Add your own text to the vocabulary
new_tokens = ["your", "custom", "tokens"]  # Add your own tokens here
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Import data
with open('./data/text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
train_data = text.split('\n')

# Initialize model
input_ids = tokenizer.encode(train_data, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
labels = input_ids.clone()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save the modified model
model.save_pretrained('modified_model')
tokenizer.save_pretrained('modified_model')

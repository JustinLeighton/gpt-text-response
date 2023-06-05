# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:23:39 2023

@author: Justin Leighton
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = 'modified_model'  # Replace with the directory where your modified model is saved
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

model.eval()
model.config.pad_token_id = model.config.eos_token_id

# Generate text
input_text = "test"  # Replace with your desired input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = input_ids.ne(tokenizer.pad_token_id).float()

# Configure generation settings
temperature = 0.8  # Controls the randomness of the generated text, lower values make it more focused
repetition_penalty = 1.2  # Adjusts the penalty for repeating the same token

output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=temperature,
    repetition_penalty=repetition_penalty,
    no_repeat_ngram_size=2
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


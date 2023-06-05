# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:55:06 2023

@author: Justin Leighton
"""

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load config
with open('config.json') as f:
    config = json.load(f)
account_sid = config['account']
auth_token = config['auth']
twilio_number = config['twilio']
your_number = config['phone']

# Create the Twilio client
client = Client(account_sid, auth_token)

# Create the Flask app
app = Flask(__name__)

# Load model
model_dir = 'modified_model'  # Replace with the directory where your modified model is saved
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model.eval()
model.config.pad_token_id = model.config.eos_token_id

# Route for receiving incoming messages
@app.route('/incoming', methods=['POST'])
def incoming_sms():
    message_body = request.form.get('Body')
    sender_number = request.form.get('From')

    # Process the incoming message
    response_body = process_incoming_message(message_body, sender_number)

    # Send a reply
    send_reply(response_body, sender_number)

    return str(MessagingResponse())

def generate_text(input_text):
    
    # Generate text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float()

    # Configure generation settings
    temperature = 0.8
    repetition_penalty = 1.2

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
    
    return generated_text

# Function to process incoming messages
def process_incoming_message(message_body, sender_number):
    print(f'Received message: {message_body} from {sender_number}')
    response_body = generate_text(message_body)
    return response_body

# Function to send a reply
def send_reply(response_body, sender_number):
    message = client.messages.create(
        body=response_body,
        from_=twilio_number,
        to=sender_number
    )
    print(f'Reply message sent to {sender_number}: {message.sid}')

if __name__ == '__main__':
    app.run()

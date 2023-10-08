 #you can create a sentiment analysis script using Transformers and a pre-trained model, such as BERT or GPT-2. Here's a simplified outline of what the Python script might look like:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input text for sentiment analysis
text = "I really enjoyed this movie. It was fantastic!"

# Tokenize and predict sentiment
tokens = tokenizer.encode(text, return_tensors="pt")
with torch.no_grad():
    output = model(tokens)
probabilities = torch.softmax(output.logits, dim=1)[0]
label = torch.argmax(probabilities).item()

# Interpret sentiment label
sentiment_labels = ["Negative", "Positive"]
print(f"Sentiment: {sentiment_labels[label]}")

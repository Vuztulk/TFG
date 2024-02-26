import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def classify_sentiment(input_text):
    # Load tokenizer and model for sentiment classification
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Perform model inference
    with torch.no_grad():
        outputs = model(input_ids)

    # Decode predicted sentiment
    predicted_class_idx = torch.argmax(outputs.logits[0]).item()
    sentiment_label = tokenizer.decode([predicted_class_idx])

    # Return formatted string
    return f'Input Text: {input_text}\nPredicted Sentiment: {sentiment_label}'
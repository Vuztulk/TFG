import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def clasificacion_sentimiento(input_text):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    
    # Tokeniza el texto de entrada
    encoded_input = tokenizer(input_text, return_tensors='pt')
    
    # Realiza la inferencia del modelo
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
    
    # Obtén la clase predicha
    predicted_class = torch.argmax(logits).item()
    
    # Define las clases de sentimiento
    sentiment_classes = ['negative', 'positive']
    
    # Devuelve el texto de entrada y la clase predicha
    return f'Input Text: {input_text}\nPredicted sentiment: {sentiment_classes[predicted_class]}'

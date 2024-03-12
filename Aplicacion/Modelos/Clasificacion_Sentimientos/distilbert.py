import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import time

def sent_distilbert_cpu(input_text):
    
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    
    encoded_input = tokenizer(input_text, return_tensors='pt')
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model(**encoded_input)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                
    predicted_class = torch.argmax(logits).item()
    sentiment_classes = ['negative', 'positive']
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time

    return sentiment_classes[predicted_class], cpu_time_str, duration

def sent_distilbert_gpu(input_text):
    return 0
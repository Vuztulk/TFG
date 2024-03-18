import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import time
import os
import psutil

def sent_sbcbi_cpu(input_text, longitud):
    
    start_time = time.time()
    
    pid = os.getpid()
    py = psutil.Process(pid)
    
    initial_memory = psutil.Process(pid).memory_info().rss
    
    tokenizer = AutoTokenizer.from_pretrained('sbcBI/sentiment_analysis_model')
    model = AutoModelForSequenceClassification.from_pretrained('sbcBI/sentiment_analysis_model')
        
    encoded_input = tokenizer(input_text, return_tensors='pt')
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model(**encoded_input)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                
    sentiment_classes = ['negative', 'neutral', 'positive']
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    final_memory = psutil.Process(pid).memory_info().rss
    memory_used = final_memory - initial_memory
    memory_used_gb = round(memory_used / (1024 * 1024 * 1024), 3)
           
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return sentiment_classes[predicted_class], cpu_time_str, formatted_duration, memory_used_gb

def sent_sbcbi_gpu(input_text, longitud):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
        
    pid = os.getpid()
    py = psutil.Process(pid)

    initial_memory = psutil.Process(pid).memory_info().rss
    
    tokenizer = AutoTokenizer.from_pretrained('sbcBI/sentiment_analysis_model')
    model = AutoModelForSequenceClassification.from_pretrained('sbcBI/sentiment_analysis_model')
    model = model.to(device)
        
    encoded_input = tokenizer(input_text, return_tensors='pt')
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    
    with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = model(**encoded_input)
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits).item()
                
    sentiment_classes = ['negative', 'neutral', 'positive']
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
        cuda_time = model_inference_event[0].cuda_time_total
        cuda_time_seconds = cuda_time / 1_000_000
        cuda_time_str = f'{cuda_time_seconds:.4f}'.replace('.', ',')
            
    final_memory = psutil.Process(pid).memory_info().rss
    memory_used = final_memory - initial_memory
    memory_used_gb = round(memory_used / (1024 * 1024 * 1024), 3)
    print(f'Memory use: {memory_used_gb} GB')   
             
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return sentiment_classes[predicted_class], cuda_time_str, formatted_duration, memory_used_gb
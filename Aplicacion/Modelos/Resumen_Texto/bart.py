import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import time

def res_bart_cpu(input_text, longitud):
    
    start_time = time.time()
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length = longitud, early_stopping=True)
                
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True), cpu_time_str, formatted_duration

def res_bart_gpu(input_text, longitud):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    start_time = time.time()
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model = model.to(device)
    
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length = longitud, early_stopping=True)
                
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
        gpu_time = model_inference_event[0].cuda_time_total
        gpu_time_seconds = gpu_time / 1_000_000
        gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True), gpu_time_str, formatted_duration
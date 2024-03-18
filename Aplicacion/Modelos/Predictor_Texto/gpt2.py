import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity
import time
import os
import psutil

def pred_gpt2_cpu(input_text, longitud):
    
    start_time = time.time()
    
    pid = os.getpid()
    py = psutil.Process(pid)
    
    initial_memory = psutil.Process(pid).memory_info().rss

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(encoded_input.shape)
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(encoded_input, max_length=longitud, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)
                
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True), cpu_time_str, formatted_duration, memory_used_gb

def pred_gpt2_gpu(input_text, longitud):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
        
    pid = os.getpid()
    py = psutil.Process(pid)

    initial_memory = psutil.Process(pid).memory_info().rss
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA,ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(input_ids, max_length=longitud, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)
 
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
        gpu_time = model_inference_event[0].cuda_time_total
        gpu_time_seconds = gpu_time / 1_000_000
        gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
              
    final_memory = psutil.Process(pid).memory_info().rss
    memory_used = final_memory - initial_memory
    memory_used_gb = round(memory_used / (1024 * 1024 * 1024), 3)
    print(f'Memory use: {memory_used_gb} GB')   
           
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return tokenizer.decode(outputs[0], skip_special_tokens=True), gpu_time_str, formatted_duration, memory_used_gb
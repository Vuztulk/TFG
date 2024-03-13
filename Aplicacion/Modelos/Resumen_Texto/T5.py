import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import psutil
import os
import time

def res_t5_cpu(input_text):
    
    start_time = time.time()
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                summary_ids = model.generate(inputs, max_length=100, min_length=30, num_beams=4, early_stopping=True)

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return summary_text, cpu_time_str, formatted_duration

def res_t5_gpu(input_text):
    return 0

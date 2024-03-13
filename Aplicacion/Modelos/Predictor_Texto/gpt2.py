import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity
import time

def pred_gpt2_cpu(input_text):
    
    start_time = time.time()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(encoded_input.shape)
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(encoded_input, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)
                
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'

    return tokenizer.decode(outputs[0], skip_special_tokens=True), cpu_time_str, formatted_duration

def pred_gpt2_gpu(input_text):
    return 0
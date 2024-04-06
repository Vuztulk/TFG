import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id

model = model.to(device)

with open('resultados.txt', 'w') as f:

    for i in range(1):
        start_time = time.time()


        with open('./input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA,ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    process_tegra = subprocess.Popen(['sudo','/usr/bin/tegrastats', '--logfile', 'tegrastats.txt','--interval','500'])
                    outputs = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)
                    process_tegra = subprocess.Popen(['/usr/bin/tegrastats', '--stop'])
                    process_tegra.terminate()

        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            gpu_time = model_inference_event[0].cuda_time_total
            gpu_time_seconds = gpu_time / 1_000_000
            gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{gpu_time_str}\n')      
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')
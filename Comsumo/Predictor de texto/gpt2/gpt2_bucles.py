import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import subprocess

# Cargar el tokenizador y el modelo
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(1):
        start_time = time.time()

        # Leer el texto de entrada desde un archivo .txt
        with open('./input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        # Codificar entrada
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape)

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    process_tegra = subprocess.Popen(['/usr/bin/tegrastats', '--logfile', 'tegrastats.txt'])
                    outputs = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)
                    process_tegra.terminate()

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')
            
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

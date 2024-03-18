import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil
import os
import time

pid = os.getpid()
py = psutil.Process(pid)

initial_memory = psutil.Process(pid).memory_info().rss

# Marcar el tiempo de inicio
start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id

# Leer el texto de entrada desde un archivo .txt
with open('Problemas\Predictor de Texto\input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape)

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Decodificar la salida
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Output text: {output_text}')


final_memory = psutil.Process(pid).memory_info().rss
memory_used = final_memory - initial_memory
memory_used_gb = round(memory_used / (1024 * 1024 * 1024), 3)
print(f'Memory use: {memory_used_gb} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

end_time = time.time()
duration = end_time - start_time
print(f'La ejecución del código tardó {duration:.4f} segundos.')

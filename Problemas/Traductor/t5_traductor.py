import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import psutil
import os

# Cargar el tokenizador y el modelo
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
input_ids = tokenizer.encode("translate English to Spanish: "+input_text, return_tensors='pt')

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Decodificar la salida
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Texto de salida: {output_text}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

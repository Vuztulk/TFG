import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
import psutil
import os
import time

def divide_input(input_text, max_length):
    words = input_text.split(' ')
    segments = []
    segment = ''
    for word in words:
        if len(segment) + len(word) + 1 > max_length:
            segments.append(segment)
            segment = ''
        segment += ' ' + word
    segments.append(segment)
    return segments

start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Dividir el texto de entrada en segmentos
segments = divide_input(input_text, max_length=200)

# Codificar y traducir cada segmento
output_text = ''
for segment in segments:
    input_ids = tokenizer.encode(segment, return_tensors='pt')
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
        output_text += tokenizer.decode(outputs[0], skip_special_tokens=True)

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(f'Texto de entrada: {input_text}\n')
print(f'Texto de salida: {output_text}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

end_time = time.time()
duration = end_time - start_time
print(f'La ejecución del código tardó {duration:.4f} segundos.')

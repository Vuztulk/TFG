import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
import psutil
import os
import concurrent.futures
import time

start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Función para realizar la inferencia del modelo
def model_inference(input_ids):
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
    return outputs

# Realizar la inferencia del modelo con el perfilador en paralelo
with concurrent.futures.ProcessPoolExecutor() as executor:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            future = executor.submit(model_inference, input_ids)
            outputs = future.result()

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Decodificar la salida
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

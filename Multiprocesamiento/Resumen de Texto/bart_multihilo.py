import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BartForConditionalGeneration, BartTokenizer
import psutil
import os
import threading
import time

start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Resumen de texto/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)

# Función para realizar la inferencia del modelo
def model_inference(inputs, outputs):
    with torch.no_grad():
        outputs[0] = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, early_stopping=True)

# Realizar la inferencia del modelo con el perfilador en paralelo
outputs = [None]
thread = threading.Thread(target=model_inference, args=(inputs, outputs))

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        thread.start()
        thread.join()

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Decodificar la salida
summary_text = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
print(f"Resumen del texto: {summary_text}")

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0] / 2. ** 30  # Uso de memoria en GB
print(f'Uso de memoria: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'Uso de CPU: {cpu_use} %')

end_time = time.time()
duration = end_time - start_time
print(f'La ejecución del código tardó {duration:.4f} segundos.')

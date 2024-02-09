import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import psutil
import os
import time

start_time = time.time()

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Predictor de Texto/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificamos la entrada
inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

# Inicializamos el perfilador de PyTorch
with torch.no_grad():
    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("model_inference"):
            summary_ids = model.generate(inputs, num_beams=4, min_length=30, max_length=100, early_stopping=True)

# Decodificamos y mostramos el resumen
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
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


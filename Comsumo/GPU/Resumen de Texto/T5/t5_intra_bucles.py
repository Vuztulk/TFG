import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import psutil
import os
import time

torch.set_num_threads(4)

# Verificar si hay una GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el tokenizador y el modelo
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(device)

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Leer el texto de entrada desde un archivo .txt
        with open('./input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        # Codificar entrada y mover a la GPU
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    summary_ids = model.generate(inputs, max_length=100, min_length=30, num_beams=4, early_stopping=True)

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            gpu_time = model_inference_event[0].cuda_time_total
            gpu_time_seconds = gpu_time / 1_000_000
            gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{gpu_time_str}\n')

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

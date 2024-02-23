import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os
import time

# Verificamos si hay una GPU disponible y, en caso afirmativo, la usamos. Si no, usamos la CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Movemos el modelo a la GPU si está disponible
model = model.to(device)

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Definimos una frase de entrada
        with open('./input.txt', 'r') as file:
            input_text = file.read().strip()

        encoded_input = tokenizer(input_text, return_tensors='pt')

        # Movemos los datos de entrada a la GPU si es necesario
        encoded_input = encoded_input.to(device)

        # Inicializamos el perfilador de PyTorch
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = model(**encoded_input)
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits).item()

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cuda_time = model_inference_event[0].cuda_time_total
            cuda_time_seconds = cuda_time / 1_000_000
            cuda_time_str = f'{cuda_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cuda_time_str}\n')

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
import psutil
import os
import time

torch.set_num_interop_threads(4)

# Función para realizar la inferencia del modelo
def model_inference(input_text):
    # Cargar el tokenizador y el modelo
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    # Codificar entrada
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(encoded_input, max_length=200, num_return_sequences=1)
    # Decodificar la salida
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Leer el texto de entrada desde un archivo .txt
        with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        # Realizar la inferencia del modelo con el perfilador
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                output_text = model_inference(input_text)

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')

        # Imprimimos la clase predicha
        #f.write(f'Texto de entrada: {input_text}\n')
        #f.write(f'Texto de salida: {output_text}\n')

        # Métricas adicionales
        #pid = os.getpid()
        #py = psutil.Process(pid)

        #memory_use = py.memory_info()[0]/2.**30  # memory use in GB
        #f.write(f'Uso de memoria: {memory_use} GB\n')

        #cpu_use = psutil.cpu_percent(interval=None)
        #f.write(f'Uso de CPU: {cpu_use} %\n')

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

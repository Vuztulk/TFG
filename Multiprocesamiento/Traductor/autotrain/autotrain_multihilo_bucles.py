import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psutil
import os
import time
import threading

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")
model = AutoModelForSeq2SeqLM.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")

# Leer el texto de entrada desde un archivo .txt
input_text = "The Great Gatsby is a classic novel by F. Scott Fitzgerald set in the Jazz Age. It follows Jay Gatsby's pursuit of Daisy Buchanan, exploring themes of love, wealth, and the American Dream. Narrated by Nick Carraway, the story delves into the complexities of society and human relationships."

# Codificar entrada
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Función para realizar la inferencia del modelo
def model_inference(input_ids, outputs):
    with torch.no_grad():
        outputs[0] = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Abrir el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutar el código 20 veces
    for i in range(10):
        start_time = time.time()

        # Realizar la inferencia del modelo con el perfilador
        outputs = [None]
        thread = threading.Thread(target=model_inference, args=(input_ids, outputs))

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                thread.start()
                thread.join()

        # Guardar las métricas del perfilador en el archivo
        cpu_time = sum([item.cpu_time_total for item in prof.key_averages()])
        cpu_time_seconds = cpu_time / 1_000_000
        f.write(f'Tiempo de CPU: {cpu_time_seconds} segundos\n')

        # Decodificar la salida
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #f.write(f'Texto de entrada: {input_text}\n')
        #f.write(f'Texto de salida: {output_text}\n')

        # Métricas adicionales
        #pid = os.getpid()
        #py = psutil.Process(pid)

        #memory_use = py.memory_info()[0]/2.**30  # memory use in GB
        #f.write(f'Memory use: {memory_use} GB\n')

        #cpu_use = psutil.cpu_percent(interval=None)
        #f.write(f'CPU use: {cpu_use} %\n')

        end_time = time.time()
        duration = end_time - start_time
        f.write(f'Duración: {duration:.4f} segundos\n\n')

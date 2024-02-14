import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os
import time
import concurrent.futures

# Función para realizar la inferencia del modelo
def model_inference(input_text):
    # Cargamos el modelo y el tokenizador preentrenados
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    encoded_input = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    return predicted_class

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Definimos una frase de entrada
        with open('/home/tfg1/TFG/Problemas/Clasificacion sentimientos/input.txt', 'r') as file:
            input_text = file.read().strip()

        # Realizamos la inferencia del modelo con el perfilador en paralelo
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    future = executor.submit(model_inference, input_text)
                    predicted_class = future.result()

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')
        # Imprimimos la clase predicha
        sentiment_classes = ['negative', 'positive']
        #f.write(f'Texto de entrada: {input_text}\n')
        #f.write(f'Sentimiento predicho: {sentiment_classes[predicted_class]}\n')

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

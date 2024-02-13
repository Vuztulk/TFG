import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os
import time
import concurrent.futures

start_time = time.time()

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Definimos una frase de entrada
with open('/home/tfg1/TFG/Problemas/Clasificacion sentimientos/input.txt', 'r') as file:
    input_text = file.read().strip()
    
encoded_input = tokenizer(input_text, return_tensors='pt')

# Función para realizar la inferencia del modelo
def model_inference(encoded_input):
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    return predicted_class

# Realizar la inferencia del modelo con el perfilador en paralelo
with concurrent.futures.ProcessPoolExecutor() as executor:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            future = executor.submit(model_inference, encoded_input)
            predicted_class = future.result()

# Imprimimos el informe del perfilador
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Imprimimos la clase predicha
sentiment_classes = ['negative', 'positive']
print(f'Input text: {input_text}')
print(f'Predicted sentiment: {sentiment_classes[predicted_class]}')

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

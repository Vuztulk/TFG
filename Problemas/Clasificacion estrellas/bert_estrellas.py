import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

with open('./Problemas/Clasificacion sentimientos/input.txt', 'r') as file:
    input_text = file.readline().strip()

# Codificamos la entrada
encoded_input = tokenizer(input_text, return_tensors='pt')

# Inicializamos el perfilador de PyTorch
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model(**encoded_input)
            logits = outputs.logits
            predicted_class = torch.argmax(logits).item()

# Imprimimos el informe del perfilador
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Imprimimos la clase predicha
sentiment_classes = ['1 estrella', '2 estrellas', '3 estrellas', '4 estrellas', '5 estrellas']
print(f'Input text: {input_text}')
print(f'Predicted sentiment: {sentiment_classes[predicted_class]}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

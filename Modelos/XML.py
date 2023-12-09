import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import XLMTokenizer, XLMModel
import psutil
import os

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
model = XLMModel.from_pretrained('xlm-mlm-en-2048')

# Definimos una frase de entrada
input_text = "What time is"
indexed_tokens = tokenizer.encode(input_text, return_tensors='pt')

# Realizamos la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model(indexed_tokens)
            predictions = outputs[0]

# Imprimimos las métricas del perfilador
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Obtenemos la predicción para la siguiente palabra
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_index])

print(f'Input text: {input_text}')
print(f'Predicted next word: {predicted_token}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')
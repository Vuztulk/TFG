import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import psutil
import os
import time

start_time = time.time()

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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

###############################################################

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

end_time = time.time()
duration = end_time - start_time
print(f'La ejecución del código tardó {duration:.4f} segundos.')
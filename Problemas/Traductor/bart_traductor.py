import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import psutil
import os

# Cargar el tokenizador y el modelo
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Leer el texto de entrada desde un archivo .txt
with open('./Problemas/Predictor de Texto/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
tokenizer.src_lang = "es_XX"
encoded_es = tokenizer(input_text, return_tensors="pt")

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            generated_tokens = model.generate(**encoded_es, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Decodificar la salida
output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(f'Output text: {output_text}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

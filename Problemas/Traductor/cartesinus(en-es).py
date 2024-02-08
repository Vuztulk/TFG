import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psutil
import os

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

# Leer el texto de entrada desde un archivo .txt
with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

input_ids = tokenizer(input_text, return_tensors="pt")

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
           generated_tokens = model.generate(**input_ids, forced_bos_token_id=tokenizer.get_lang_id("es"))

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(f'Texto de entrada: {input_text}\n')
print(f'Texto de salida: {output_text}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

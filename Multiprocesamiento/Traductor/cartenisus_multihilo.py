import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psutil
import os
import threading
import time

start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

# Leer el texto de entrada desde un archivo .txt
#with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
    #input_text = file.read().replace('\n', '')
input_text = "The Great Gatsby is a classic novel by F. Scott Fitzgerald set in the Jazz Age. It follows Jay Gatsby's pursuit of Daisy Buchanan, exploring themes of love, wealth, and the American Dream. Narrated by Nick Carraway, the story delves into the complexities of society and human relationships."

input_ids = tokenizer(input_text, return_tensors="pt")

# Función para realizar la inferencia del modelo
def model_inference(input_ids, generated_tokens):
    with torch.no_grad():
        generated_tokens[0] = model.generate(**input_ids, forced_bos_token_id=tokenizer.get_lang_id("es"))

# Realizar la inferencia del modelo con el perfilador en paralelo
generated_tokens = [None]
thread = threading.Thread(target=model_inference, args=(input_ids, generated_tokens))

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        thread.start()
        thread.join()

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

output_text = tokenizer.batch_decode(generated_tokens[0], skip_special_tokens=True)[0]

print(f'Texto de entrada: {input_text}\n')
print(f'Texto de salida: {output_text}')

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

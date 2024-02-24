import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import psutil
import os
import time

class TextDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.text = file.read().replace('\n', '').split('. ')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

# Cargar el tokenizador y el modelo
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Crear un DataLoader con paralelización a nivel de datos
dataset = TextDataset('/home/tfg1/TFG/Problemas/Resumen de texto/input.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Ejecutar el código una vez
start_time = time.time()

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for i, input_text in enumerate(dataloader):
                inputs = tokenizer.encode("summarize: " + input_text[0], return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = model.generate(inputs, max_length=100, min_length=30, num_beams=4, early_stopping=True)

summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Calcular el tiempo de CPU y total
end_time = time.time()
duration = end_time - start_time
cpu_time = prof.key_averages().total_cpu_time
cpu_time_seconds = cpu_time / 1_000_000
cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
duration_str = f'{duration:.4f}'.replace('.', ',')
print(f'Tiempo de CPU: {cpu_time_str} segundos')
print(f'Tiempo total: {duration_str} segundos')


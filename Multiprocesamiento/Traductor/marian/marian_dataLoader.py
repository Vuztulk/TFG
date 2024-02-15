import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
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

start_time = time.time()

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Crear un DataLoader con paralelización a nivel de datos
dataset = TextDataset('/home/tfg1/TFG/Multiprocesamiento/Traductor/marian/input.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Inicializar una lista para guardar las traducciones
translations = []

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for i, input_text in enumerate(dataloader):
                input_ids = tokenizer.encode(input_text[0], return_tensors='pt')
                outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translations.append((input_text[0], output_text))

# Unir todas las partes del texto de entrada y salida
input_text = '. '.join([text for text, _ in translations])
output_text = '. '.join([text for _, text in translations])

# Imprimir el texto de entrada y salida
print(f'Texto de entrada: {input_text}\n')
print(f'Texto de salida: {output_text}\n')

# Imprimir las métricas del perfilador
print("Métricas del perfilador:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

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

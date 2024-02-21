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

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Crear un DataLoader con paralelización a nivel de datos
dataset = TextDataset('/home/tfg1/TFG/Problemas/Traductor/input.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Abrir el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutar el código 10 veces
    for _ in range(10):
        start_time = time.time()

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for i, input_text in enumerate(dataloader):
                        input_ids = tokenizer.encode(input_text[0], return_tensors='pt')
                        outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
                        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        #print(f'Texto de entrada: {input_text[0]}\n')
                        #print(f'Texto de salida: {output_text}\n')

        # Guardar las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

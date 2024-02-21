import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os
import time

class TextDataset(Dataset):
    def __init__(self, filename, tokenizer):
        with open(filename, 'r') as file:
            self.text = file.read().strip().split('. ')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        input_text = self.text[idx]
        encoded_input = self.tokenizer(input_text, return_tensors='pt')
        return encoded_input

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained('sbcBI/sentiment_analysis_model')
model = AutoModelForSequenceClassification.from_pretrained('sbcBI/sentiment_analysis_model')

# Creamos un DataLoader con paralelización a nivel de datos
dataset = TextDataset('/home/tfg1/TFG/Problemas/Clasificacion sentimientos/input.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Inicializamos el perfilador de PyTorch
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for encoded_input in dataloader:
                        outputs = model(**encoded_input)
                        logits = outputs.logits
                        predicted_class = torch.argmax(logits).item()

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

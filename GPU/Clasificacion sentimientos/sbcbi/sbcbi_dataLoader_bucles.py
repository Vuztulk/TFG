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
        return {key: val.squeeze(0) for key, val in encoded_input.items()}

# Verificamos si hay una GPU disponible y, en caso afirmativo, la usamos. Si no, usamos la CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargamos el modelo y el tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained('sbcBI/sentiment_analysis_model')
model = AutoModelForSequenceClassification.from_pretrained('sbcBI/sentiment_analysis_model')

# Movemos el modelo a la GPU si está disponible
model = model.to(device)

# Creamos un DataLoader con paralelización a nivel de datos
dataset = TextDataset('./input.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Inicializamos el perfilador de PyTorch
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for encoded_input in dataloader:
                        # Movemos los datos de entrada a la GPU si es necesario
                        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
                        outputs = model(**encoded_input)
                        logits = outputs.logits
                        predicted_class = torch.argmax(logits).item()

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cuda_time = model_inference_event[0].cuda_time_total
            cuda_time_seconds = cuda_time / 1_000_000
            cuda_time_str = f'{cuda_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cuda_time_str}\n')

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

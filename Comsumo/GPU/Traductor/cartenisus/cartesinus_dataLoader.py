import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time
from torch.utils.data import Dataset, DataLoader

# Define una clase para el conjunto de datos
class TextDataset(Dataset):
    def __init__(self, filename, tokenizer):
        with open(filename, 'r') as file:
            self.text = file.read().replace('\n', '')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.tokenizer([self.text[idx]], return_tensors='pt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = model.to(device)

# Crear el DataLoader
dataset = TextDataset('./input.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Abrir el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutar el código 10 veces
    for i in range(10):
        start_time = time.time()

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for input_batch in dataloader:
                        input_batch = {key: tensor.to(device) for key, tensor in input_batch.items()}
                        generated_tokens = model.generate(**input_batch, forced_bos_token_id=tokenizer.get_lang_id("es"))
        
        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            gpu_time = model_inference_event[0].cuda_time_total
            gpu_time_seconds = gpu_time / 1_000_000
            gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{gpu_time_str}\n')

        output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

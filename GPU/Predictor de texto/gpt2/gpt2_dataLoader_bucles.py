import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

class TextDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.text = file.read().replace('\n', '').split('. ')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

# Verificamos si hay una GPU disponible y, en caso afirmativo, la usamos. Si no, usamos la CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el tokenizador y el modelo
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id

# Movemos el modelo a la GPU
model = model.to(device)

# Crear un DataLoader con paralelización a nivel de datos
dataset = TextDataset('./input.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Abrir el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutar el código 10 veces
    for _ in range(10):
        start_time = time.time()

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for i, input_text in enumerate(dataloader):
                        input_ids = tokenizer.encode(input_text[0], return_tensors='pt').to(device)
                        attention_mask = torch.ones(input_ids.shape, device=device)
                        outputs = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1, do_sample=True, attention_mask=attention_mask)

        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            gpu_time = model_inference_event[0].cuda_time_total
            gpu_time_seconds = gpu_time / 1_000_000
            gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{gpu_time_str}\n')

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

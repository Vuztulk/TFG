import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import time

# Define una clase para el conjunto de datos
class TextDataset(Dataset):
    def __init__(self, filename, tokenizer):
        with open(filename, 'r') as file:
            self.text = file.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.text[idx], return_tensors='pt')
        return {key: tensor.squeeze() for key, tensor in tokenized.items()}

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

# Crear el DataLoader
dataset = TextDataset('/home/tfg1/TFG/Problemas/Traductor/input.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Ejecutar el código una vez
start_time = time.time()

# Realizar la inferencia del modelo con el perfilador
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for input_batch in dataloader:
                generated_tokens = model.generate(input_ids=input_batch['input_ids'], forced_bos_token_id=tokenizer.get_lang_id("es"))


model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
if model_inference_event:
    cpu_time = model_inference_event[0].cpu_time_total
    cpu_time_seconds = cpu_time / 1_000_000
    cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
    print(f'Tiempo de CPU: {cpu_time_str} segundos')

output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Calcular el tiempo de CPU y total
end_time = time.time()
duration = end_time - start_time
duration_str = f'{duration:.4f}'.replace('.', ',')
print(f'Tiempo total: {duration_str} segundos')



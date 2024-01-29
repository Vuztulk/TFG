import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sentencepiece as spm
import psutil
import os

# Cargamos el modelo preentrenado
model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='sentencepiece')
model.eval()

# Cargamos el modelo de SentencePiece
sp = spm.SentencePieceProcessor()
sp.Load('./sentencepiece.bpe.model')

# Definimos la frase a traducir
sentence = "The quick brown fox jumps over the lazy dog"

# Aplicamos SentencePiece a la frase
tokens = sp.EncodeAsPieces(sentence)
input_tensor = torch.tensor([model.encode(tokens)])

# Inicializamos el perfilador de PyTorch
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model.generate(input_tensor)

# Imprimimos el informe del perfilador
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Obtenemos la traducción
translation = model.decode(outputs[0])

print(f'Traducción: {translation}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Uso de memoria: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'Uso de CPU: {cpu_use} %')

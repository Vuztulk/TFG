import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Cargar el tokenizador y el modelo
tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')
model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')

# Leer el texto de entrada desde un archivo .txt
with open('./Problemas/Predictor de Texto/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

# Realizar la inferencia del modelo
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Decodificar la salida
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Texto de salida: {output_text}')

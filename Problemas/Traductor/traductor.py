import torch
from transformers import MarianMTModel, MarianTokenizer

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Leer el texto de entrada desde un archivo .txt
with open('./Problemas/Predictor de Texto/input.txt', 'r') as file:
    input_text = file.read().replace('\n', '')

# Codificar entrada
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Realizar la inferencia del modelo
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Decodificar la salida
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Texto de salida: {output_text}')

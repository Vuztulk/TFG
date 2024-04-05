import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
import subprocess
import time

# Función para realizar la inferencia del modelo
def model_inference(input_text):
    # Cargar el tokenizador y el modelo
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    # Codificar entrada
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(encoded_input, max_length=200, num_return_sequences=1)
    # Decodificar la salida
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(1):
        start_time = time.time()

        # Leer el texto de entrada desde un archivo .txt
        with open('./input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        # Realizar la inferencia del modelo con el perfilador
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                process_tegra = subprocess.Popen(['/usr/bin/tegrastats', '--logfile', 'tegrastats.txt'])
                output_text = model_inference(input_text)
                process_tegra.terminate()

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

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer
import time
from multiprocessing import Pool

def process_text(input_text):
    # Codificar entrada
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Realizar la inferencia del modelo con el perfilador
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

    # Decodificar la salida
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Obtener el tiempo del perfilador
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')

    return output_text, cpu_time_str

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# Leer el texto de entrada desde un archivo .txt
input_text = "En un lugar de la mancha, de cuyo nombre no quiero acordarme. no ha mucho tiempo que vivia un hidalgo de los de lanza en astillero.adarga antigua, rocin flaco y galgo corredor."

# Dividir el texto de entrada en partes
parts = input_text.split('.')

# Abrir el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutar el código 10 veces
    for _ in range(2):
        start_time = time.time()

        # Crear un pool de procesos
        with Pool(processes=len(parts)) as pool:
            results = pool.map(process_text, parts)

        # Sumar los tiempos de todos los procesos
        total_profiler_time = sum([float(result[1].replace(',', '.')) for result in results])

        # Guardar los resultados
        f.write(f'{total_profiler_time:.4f}\n'.replace('.', ','))


        end_time = time.time()
        duration = end_time - start_time
        f.write(f'{duration:.4f}\n'.replace('.', ','))

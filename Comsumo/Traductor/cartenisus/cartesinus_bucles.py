import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import subprocess
import time

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

# Abrimos el archivo de resultados
with open('resultados.txt', 'w') as f:
    # Ejecutamos el código 10 veces
    for i in range(1):
        start_time = time.time()

        # Leer el texto de entrada desde un archivo .txt
        with open('/home/tfg1/TFG/Problemas/Traductor/input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        # Codificar entrada
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Realizar la inferencia del modelo con el perfilador
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    # Ejecutar el comando justo antes de la inferencia
                    process = subprocess.Popen(['sudo', '/usr/bin/tegrastats'], stdout=subprocess.PIPE)
                    output_before, _ = process.communicate()

                    # Realizar la inferencia
                    generated_tokens = model.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.get_lang_id("es"))

                    # Ejecutar el comando justo después de la inferencia
                    process = subprocess.Popen(['sudo', '/usr/bin/tegrastats'], stdout=subprocess.PIPE)
                    output_after, _ = process.communicate()

                    # Ahora output_before y output_after son objetos Python (bytes) que contienen la salida del comando
                    # Puedes convertirlos a una cadena si lo necesitas
                    output_before_str = output_before.decode('utf-8')
                    output_after_str = output_after.decode('utf-8')
        
        # Guardamos las métricas del perfilador en el archivo
        model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
        if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            f.write(f'{cpu_time_str}\n')

        output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        end_time = time.time()
        duration = end_time - start_time
        duration_str = f'{duration:.4f}'.replace('.', ',')
        f.write(f'{duration_str}\n')

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def trad_autotrain_cpu(input_text):
    
    start_time = time.time()

    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")
    model = AutoModelForSeq2SeqLM.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")

    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    # Realizar la inferencia del modelo con el perfilador
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return output_text, cpu_time_str, formatted_duration

def trad_autotrain_gpu(input_text):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")
    model = AutoModelForSeq2SeqLM.from_pretrained("robertrengel/autotrain-traductor-en-es-2023-3608896666")
    model = model.to(device)
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Realizar la inferencia del modelo con el perfilador
    with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
        gpu_time = model_inference_event[0].cuda_time_total
        gpu_time_seconds = gpu_time / 1_000_000
        gpu_time_str = f'{gpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')

    return output_text, gpu_time_str, formatted_duration
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def trad_cartenisus_cpu(input_text):
    
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

    input_ids = tokenizer(input_text, return_tensors="pt")

    # Realizar la inferencia del modelo con el perfilador
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                generated_tokens = model.generate(**input_ids, forced_bos_token_id=tokenizer.get_lang_id("es"))

    output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    model_inference_event = [item for item in prof.key_averages() if item.key == "model_inference"]
    if model_inference_event:
            cpu_time = model_inference_event[0].cpu_time_total
            cpu_time_seconds = cpu_time / 1_000_000
            cpu_time_str = f'{cpu_time_seconds:.4f}'.replace('.', ',')
            
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = f'{duration:.4f}'.replace('.', ',')
    
    return output_text, cpu_time_str, formatted_duration

def trad_cartenisus_gpu(input_text):
    return 0
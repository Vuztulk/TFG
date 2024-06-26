import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import subprocess
import time

tokenizer = AutoTokenizer.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("cartesinus/iva_mt_wslot-m2m100_418M-en-es")

with open('resultados.txt', 'w') as f:

    for i in range(1):
        start_time = time.time()

        with open('./input.txt', 'r') as file:
            input_text = file.read().replace('\n', '')

        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                process_tegra = subprocess.Popen(['sudo','/usr/bin/tegrastats', '--logfile', 'tegrastats.txt','--interval','500'])
                generated_tokens = model.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.get_lang_id("es"))
                process_tegra = subprocess.Popen(['/usr/bin/tegrastats', '--stop'])
                process_tegra.terminate()

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

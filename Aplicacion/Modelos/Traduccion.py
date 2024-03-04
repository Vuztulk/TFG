import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import MarianMTModel, MarianTokenizer

def translate_text(input_text):
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return f'Texto de entrada: {input_text}\nTexto de salida: {output_text}'

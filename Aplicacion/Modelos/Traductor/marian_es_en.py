import torch
from transformers import MarianMTModel, MarianTokenizer

def marian(input_text):

    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text
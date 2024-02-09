import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import os
import threading
import time

start_time = time.time()

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sbcBI/sentiment_analysis_model')
model = AutoModelForSequenceClassification.from_pretrained('sbcBI/sentiment_analysis_model')

# Define an input phrase
with open('/home/tfg1/TFG/Problemas/Clasificacion sentimientos/input.txt', 'r') as file:
    input_text = file.readline().strip()
    
encoded_input = tokenizer(input_text, return_tensors='pt')

# Function to perform model inference
def model_inference(encoded_input, outputs):
    with torch.no_grad():
        outputs[0] = model(**encoded_input)
        logits = outputs[0].logits
        predicted_class = torch.argmax(logits).item()
        outputs[1] = predicted_class

# Perform model inference with the profiler in parallel
outputs = [None, None]
thread = threading.Thread(target=model_inference, args=(encoded_input, outputs))

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        thread.start()
        thread.join()

# Print the profiler metrics
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Print the predicted class
sentiment_classes = ['negative', 'neutral', 'positive']
print(f'Input text: {input_text}')
print(f'Predicted sentiment: {sentiment_classes[outputs[1]]}')

# Additional metrics
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

end_time = time.time()
duration = end_time - start_time
print(f'The execution of the code took {duration:.4f} seconds.')

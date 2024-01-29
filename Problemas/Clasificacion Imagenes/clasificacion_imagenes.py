import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import models, transforms
from PIL import Image
import psutil
import os
import json

# Cargamos el modelo preentrenado
model = models.resnet50(pretrained=True)
model.eval()

# Definimos las transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargamos y transformamos la imagen
image = Image.open('./Problemas/Clasificacion Imagenes/Imagenes/Golden Retriever.jpg')
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Inicializamos el perfilador de PyTorch
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model(input_batch)

# Imprimimos el informe del perfilador
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Cargamos las etiquetas de clase desde el archivo JSON
with open('./Problemas/Clasificacion Imagenes/imagenet-simple-labels.json') as f:
    labels = json.load(f)

# Definimos una función para obtener la etiqueta de la clase
def class_id_to_label(i):
    return labels[i]

# Obtenemos la clase predicha
_, predicted_idx = torch.max(outputs, 1)
predicted_class = class_id_to_label(predicted_idx.item())

print(f'Clase predicha: {predicted_class}')

# Métricas adicionales
pid = os.getpid()
py = psutil.Process(pid)

memory_use = py.memory_info()[0]/2.**30  # memory use in GB
print(f'Memory use: {memory_use} GB')

cpu_use = psutil.cpu_percent(interval=None)
print(f'CPU use: {cpu_use} %')

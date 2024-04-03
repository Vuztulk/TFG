import re

power_values_cpu = []
power_values_sys = []
power_values_gpu = []

with open('Problemas/tegrastats', 'r') as file:
    for line in file:
        match_cpu = re.search(r'VDD_CPU_CV (\d+)mW', line)
        match_sys = re.search(r'VIN_SYS_5V0 (\d+)mW', line)
        match_gpu = re.search(r'VDD_GPU_SOC (\d+)mW', line)

        if match_cpu:
            power_values_cpu.append(int(match_cpu.group(1)) / 1000)  # Convertir de mW a W
        
        if match_sys:
            power_values_sys.append(int(match_sys.group(1)) / 1000)  # Convertir de mW a W
        
        if match_gpu:
            power_values_gpu.append(int(match_gpu.group(1)) / 1000)  # Convertir de mW a W

# Calcular el consumo medio y máximo de energía para CPU, SYS y GPU
average_power_cpu = sum(power_values_cpu) / len(power_values_cpu) if power_values_cpu else 0
max_cpu = max(power_values_cpu) if power_values_cpu else 0

average_power_sys = sum(power_values_sys) / len(power_values_sys) if power_values_sys else 0
max_sys = max(power_values_sys) if power_values_sys else 0

average_power_gpu = sum(power_values_gpu) / len(power_values_gpu) if power_values_gpu else 0
max_gpu = max(power_values_gpu) if power_values_gpu else 0

# Imprimir los valores en el formato requerido
print(f'{average_power_cpu:.2f} - {max_cpu:.2f} ')
print(f'{average_power_sys:.2f} - {max_sys:.2f} ')
print(f'{average_power_gpu:.2f} - {max_gpu:.2f} ')

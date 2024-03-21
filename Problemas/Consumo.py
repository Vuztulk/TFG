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
            power_values_cpu.append(int(match_cpu.group(1)))
        
        if match_sys:
            power_values_sys.append(int(match_sys.group(1)))
        
        if match_gpu:
            power_values_gpu.append(int(match_gpu.group(1)))

# Calcular el consumo medio y máximo de energía para CPU, SYS y GPU
average_power_cpu = sum(power_values_cpu) / len(power_values_cpu) if power_values_cpu else 0
max_cpu = max(power_values_cpu) if power_values_cpu else 0

average_power_sys = sum(power_values_sys) / len(power_values_sys) if power_values_sys else 0
max_sys = max(power_values_sys) if power_values_sys else 0

average_power_gpu = sum(power_values_gpu) / len(power_values_gpu) if power_values_gpu else 0
max_gpu = max(power_values_gpu) if power_values_gpu else 0

# Imprimir los resultados
print(f'El consumo medio de energía de CPU: {average_power_cpu} mW')
print(f'El consumo máximo de energía de CPU: {max_cpu} mW')

print(f'El consumo medio de energía de SYS: {average_power_sys} mW')
print(f'El consumo máximo de energía de SYS: {max_sys} mW')

print(f'El consumo medio de energía de GPU: {average_power_gpu} mW')
print(f'El consumo máximo de energía de GPU: {max_gpu} mW')

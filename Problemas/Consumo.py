import re

with open('tegrastats.log', 'r') as f:
    lines = f.readlines()

power_values_cpu = [float(re.search('VDD_CPU_SOC (\d+)mW', line).group(1)) for line in lines]

power_values_sys = [float(re.search('VIN_SYS_5V0 (\d+)mW', line).group(1)) for line in lines]

power_values_gpu = [float(re.search('VDD_GPU_SOC (\d+)mW', line).group(1)) for line in lines]

average_power = sum(power_values_cpu) / len(power_values_cpu)

average_power = sum(power_values_sys) / len(power_values_sys)

average_power = sum(power_values_gpu) / len(power_values_gpu)

print(f'El consumo medio de energía de CPU: {power_values_cpu} mW')

print(f'El consumo medio de energía de SYS: {power_values_sys} mW')

print(f'El consumo medio de energía de GPU: {power_values_gpu} mW')
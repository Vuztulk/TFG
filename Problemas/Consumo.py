import re

with open('tegrastats.log', 'r') as f:
    lines = f.readlines()

power_values_cpu = [float(re.search('VDD_CPU_SOC (\d+)mW', line).group(1)) for line in lines]

power_values_sys = [float(re.search('VIN_SYS_5V0 (\d+)mW', line).group(1)) for line in lines]

power_values_gpu = [float(re.search('VDD_GPU_SOC (\d+)mW', line).group(1)) for line in lines]

average_power_cpu = sum(power_values_cpu) / len(power_values_cpu)
max_cpu = max(power_values_cpu)

average_power_sys = sum(power_values_sys) / len(power_values_sys)
max_sys = max(power_values_sys)

average_power_gpu = sum(power_values_gpu) / len(power_values_gpu)
max_gpu = max(power_values_gpu)

print(f'El consumo medio de energía de CPU: {average_power_cpu} mW')
print(f'El consumo maximo de energía de CPU: {max_cpu} mW')


print(f'El consumo medio de energía de SYS: {average_power_sys} mW')
print(f'El consumo maximo de energía de SYS: {max_sys} mW')


print(f'El consumo medio de energía de GPU: {average_power_gpu} mW')
print(f'El consumo maximo de energía de GPU: {max_gpu} mW')

import numpy as np
from scipy.signal import hilbert
from cca import CCA,FB_CCA,ECCA,FB_ECCA


import numpy as np

# 生成原始信号
def generate_signal(length, sample_rate):
    time = np.arange(length) / sample_rate
    signal = np.sin(2 * np.pi * 50 * time)  # 假设原始信号为50Hz的正弦波
    return signal

# 生成谐波信号
def generate_harmonics(signal, harmonics):
    harmonics_signal = np.zeros_like(signal)
    for i in range(harmonics):
        frequency = (i + 2) * 50  # 计算谐波的频率，假设基频为50Hz
        harmonic = np.sin(2 * np.pi * frequency * time)  # 生成谐波信号
        harmonics_signal += harmonic
    return harmonics_signal

# 使用最小二乘法拟合谐波信号的参数
def fit_harmonics(signal, harmonics):
    time = np.arange(len(signal))
    A = np.column_stack([np.sin(2 * np.pi * (i + 2) * 50 * time) for i in range(harmonics)])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    return coeffs

# 参数
length = 1000  # 信号长度
sample_rate = 250  # 采样率
harmonics = 4  # 谐波次数
time = np.arange(length) / sample_rate
# 生成原始信号
signal = generate_signal(length, sample_rate)

# 生成谐波信号
harmonics_signal = generate_harmonics(signal, harmonics)

# 拟合谐波信号的参数
coeffs = fit_harmonics(signal, harmonics)

# 合成信号
synthesized_signal = np.zeros_like(signal)
for i in range(harmonics):
    frequency = (i + 2) * 50  # 计算谐波的频率，假设基频为50Hz
    harmonic = coeffs[i] * np.sin(2 * np.pi * frequency * time)  # 生成谐波信号
    synthesized_signal += harmonic

# 打印拟合的谐波参数
for i, coeff in enumerate(coeffs):
    print(f"Harmonic {i + 2}: Amplitude = {coeff}")

# 绘制原始信号和合成信号
import matplotlib.pyplot as plt
# 生成时间轴

plt.figure(figsize=(10, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time, synthesized_signal, label='Synthesized Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

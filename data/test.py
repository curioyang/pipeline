# import numpy as np
# import matplotlib.pyplot as plt
# import librosa  # 需要安装 librosa 库
# import librosa.display
#
# # 加载音频文件
# file_path = "output1.wav"  # 替换为你的音频文件路径
# y, sr = librosa.load(file_path, sr=16000)  # 默认采样率 16000 Hz
#
# # 计算 STFT
# n_fft = 400  # FFT 窗口大小
# hop_length = 160  # 窗口移动的步长
# window = "hann"  # 使用汉宁窗
#
# D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
# #for i in D:
# #    print(i)
# print(D[13])
# # 提取幅值和相位
# magnitude, phase = librosa.magphase(D)
# magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)  # 转换为对数尺度（以分贝为单位）
#
# # 可视化频谱图
#
# # 可视化频谱图
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(
#     magnitude_db,
#     sr=sr,
#     hop_length=hop_length,
#     x_axis='time',
#     y_axis='log',
#     cmap='viridis'
# )
# plt.colorbar(format="%+2.0f dB")
# plt.title("STFT 频谱图")
# plt.xlabel("时间 (秒)")
# plt.ylabel("频率 (Hz)")
# plt.tight_layout()
# plt.show()

# import numpy as np


# def cosine_similarity(a, b):
#     # 计算点积
#     dot_product = np.dot(a, b)
#     # 计算向量的模长
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     # 计算余弦相似度
#     return dot_product / (norm_a * norm_b)

# pythondata = np.load("window.npy")
# cppdata = []
# with open("window.txt", "r") as file:
#     for line in file:
#         for i in line.split(" ")[:-1]:
#             cppdata.append(float(i))

# b = np.array(cppdata)
# print(cosine_similarity(b, pythondata))
# # compute cosine with cppdata and pythondata

# import numpy as np

# data = np.fromfile("wte_weights.bin", dtype=np.float32).astype(np.bfloat16)
# print(data)
# data.tofile("wte_weights_bf16.bin")


import torch

def load_fp32_bin(file_path):
   
    # 根据元素数量读取文件
    with open(file_path, 'rb') as f:
        data = torch.frombuffer(f.read(), dtype=torch.float32)
    
    return data

# 示例使用
input_file_path = 'wte_weights.bin'  # 输入文件路径
output_file_path = 'wte_weights_bf16.bin'  # 输出文件路径
tensor_fp32 = load_fp32_bin(input_file_path)
tensor_bf16 = tensor_fp32.bfloat16()

# 定义输出文件路径
output_file_path = 'wte_weights_bf16.bin'

# 将 BF16 张量的数据写入文件
with open(output_file_path, 'wb') as file:
    # 先将 BF16 张量转换为 CPU 上的连续内存布局，然后获取其字节数据
    byte_data = tensor_bf16.cpu().contiguous().storage().data_ptr()
    # 获取字节长度
    num_bytes = tensor_bf16.numel() * tensor_bf16.element_size()
    # 使用 ctypes 来复制内存中的数据并写入文件
    import ctypes
    ctypes_array = (ctypes.c_char * num_bytes).from_address(byte_data)
    file.write(ctypes_array)
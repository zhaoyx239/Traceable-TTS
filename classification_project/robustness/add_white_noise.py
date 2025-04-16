import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def add_white_noise(input_path, output_path, noise_level=0.01):
    """
    读取音频文件，添加白噪声，并保存到指定路径。
    
    参数:
    - input_path: 输入音频文件的路径
    - output_path: 输出音频文件的路径
    - noise_level: 噪声水平（默认值为 0.01，表示噪声的强度）
    """
    # 加载音频文件
    y, sr = librosa.load(input_path, sr=None)
    
    # 生成白噪声
    noise = np.random.normal(0, noise_level, len(y))
    
    # 将噪声添加到音频信号中
    y_noisy = y + noise
    
    # 保存添加噪声后的音频文件
    sf.write(output_path, y_noisy, sr)

def get_all_wav_files(input_dir):
    """
    获取输入目录下所有 .wav 文件的路径，并存储在列表中。
    
    参数:
    - input_dir: 输入目录路径
    
    返回:
    - wav_files: 所有 .wav 文件的路径列表
    """
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def process_files(wav_files, input_dir, output_dir):
    """
    处理所有 .wav 文件，为每个文件添加白噪声，并保存到输出目录。
    
    参数:
    - wav_files: 所有 .wav 文件的路径列表
    - input_dir: 输入目录路径
    - output_dir: 输出目录路径
    """
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # 计算相对路径，保持目录结构
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # 创建对应的输出目录
        output_file_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_file_dir, exist_ok=True)
        
        # 生成输出文件的路径
        output_file_path = os.path.join(output_file_dir, os.path.basename(input_file_path))
        
        # 添加白噪声并保存
        add_white_noise(input_file_path, output_file_path)

if __name__ == "__main__":
    # 输入目录
    input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/test-clean"
    
    # 输出目录
    output_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS-noise/test-clean"
    
    # 获取所有 .wav 文件路径
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    # 处理所有文件
    process_files(wav_files, input_dir, output_dir)
    
    print("All files processed and saved.")
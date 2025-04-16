import os
import librosa
import soundfile as sf
from tqdm import tqdm

def change_speed(input_path, output_path, speed_factor):
    """
    读取音频文件，改变其速度，并保存到指定路径。
    
    参数:
    - input_path: 输入音频文件的路径
    - output_path: 输出音频文件的路径
    - speed_factor: 速度变化因子
    """
    # 加载音频文件
    y, sr = librosa.load(input_path, sr=None)
    
    # 使用 librosa 的 time_stretch 函数改变速度
    y_stretched = librosa.effects.time_stretch(y, rate = speed_factor)
    
    # 保存变速后的音频文件
    sf.write(output_path, y_stretched, sr)

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

def process_files(wav_files, input_dir, output_dir_1_2, output_dir_0_8):
    """
    处理所有 .wav 文件，分别生成 1.2 倍速和 0.8 倍速的音频文件。
    
    参数:
    - wav_files: 所有 .wav 文件的路径列表
    - input_dir: 输入目录路径
    - output_dir_1_2: 1.2 倍速的输出目录路径
    - output_dir_0_8: 0.8 倍速的输出目录路径
    """
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # 计算相对路径，保持目录结构
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # 创建对应的输出目录
        output_dir_1_2_path = os.path.join(output_dir_1_2, relative_path)
        output_dir_0_8_path = os.path.join(output_dir_0_8, relative_path)
        
        os.makedirs(output_dir_1_2_path, exist_ok=True)
        os.makedirs(output_dir_0_8_path, exist_ok=True)
        
        # 生成输出文件的路径
        output_file_1_2 = os.path.join(output_dir_1_2_path, os.path.basename(input_file_path))
        output_file_0_8 = os.path.join(output_dir_0_8_path, os.path.basename(input_file_path))
        
        # 处理 1.2 倍速
        change_speed(input_file_path, output_file_1_2, 1.2)
        
        # 处理 0.8 倍速
        change_speed(input_file_path, output_file_0_8, 0.8)

if __name__ == "__main__":
    # 输入目录
    input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/it3/test-clean"
    
    # 输出目录
    output_dir_1_2 = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/1.2/test-clean"
    output_dir_0_8 = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/0.8/test-clean"
    
    # 获取所有 .wav 文件路径
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    # 处理所有文件
    process_files(wav_files, input_dir, output_dir_1_2, output_dir_0_8)
    
    print("All files processed and saved.")
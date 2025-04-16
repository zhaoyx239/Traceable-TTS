import os
import librosa
import soundfile as sf
from tqdm import tqdm

def volume_scale(input_path, output_path, scale_factor=0.5):
    """
    对音频进行音量变化，并保存到指定路径。
    
    参数:
    - input_path: 输入音频文件的路径
    - output_path: 输出音频文件的路径
    - scale_factor: 音量缩放因子（0.5 表示音量减半，2.0 表示音量加倍）
    """
    y, sr = librosa.load(input_path, sr=None)
    y_scaled = y * scale_factor
    sf.write(output_path, y_scaled, sr)

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
    处理所有 .wav 文件，进行音量变化，并保存到输出目录。
    
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
        
        # 处理音量变化
        volume_scale(input_file_path, output_file_path, scale_factor=0.5)

if __name__ == "__main__":
    # 输入目录
    input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/it3/test-clean"
    
    # 输出目录
    output_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/volume/test-clean"
    
    # 获取所有 .wav 文件路径
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    # 处理所有文件
    process_files(wav_files, input_dir, output_dir)
    
    print("All files processed and saved.")
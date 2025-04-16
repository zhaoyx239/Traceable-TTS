import os
import subprocess
from tqdm import tqdm

def convert_format(input_path, output_path, target_format="mp3"):
    """
    将音频文件转换为指定格式，并保存到指定路径。
    
    参数:
    - input_path: 输入音频文件的路径
    - output_path: 输出音频文件的路径
    - target_format: 目标格式（如 "mp3", "wav"）
    """
    if target_format == "mp3":
        codec = "libmp3lame"
    elif target_format == "wav":
        codec = "pcm_s16le"  # WAV 格式的编码器
    else:
        raise ValueError("Unsupported target format")
    
    subprocess.run(["ffmpeg", "-i", input_path, "-codec:a", codec, output_path])

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

def process_files(wav_files, input_dir, output_dir_mp3, output_dir_wav):
    """
    处理所有 .wav 文件，进行格式转换，并保存到输出目录。
    
    参数:
    - wav_files: 所有 .wav 文件的路径列表
    - input_dir: 输入目录路径
    - output_dir_mp3: MP3 格式的输出目录路径
    - output_dir_wav: WAV 格式的输出目录路径
    """
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # 计算相对路径，保持目录结构
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # 创建对应的输出目录
        output_dir_mp3_path = os.path.join(output_dir_mp3, relative_path)
        output_dir_wav_path = os.path.join(output_dir_wav, relative_path)
        
        os.makedirs(output_dir_mp3_path, exist_ok=True)
        os.makedirs(output_dir_wav_path, exist_ok=True)
        
        # 生成输出文件的路径
        output_file_mp3 = os.path.join(output_dir_mp3_path, os.path.basename(input_file_path).replace(".wav", ".mp3"))
        output_file_wav = os.path.join(output_dir_wav_path, os.path.basename(input_file_path))
        
        # 处理格式转换：WAV -> MP3
        convert_format(input_file_path, output_file_mp3, target_format="mp3")
        
        # 处理格式转换：MP3 -> WAV
        convert_format(output_file_mp3, output_file_wav, target_format="wav")

if __name__ == "__main__":
    # 输入目录
    input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/it3/test-clean"
    
    # 输出目录
    output_dir_mp3 = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/format-mp3/test-clean"
    output_dir_wav = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/format-wav/test-clean"
    
    # 获取所有 .wav 文件路径
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    # 处理所有文件
    process_files(wav_files, input_dir, output_dir_mp3, output_dir_wav)
    
    print("All files processed and saved.")
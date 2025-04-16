import os
import subprocess
from tqdm import tqdm

def convert_format(input_path, output_path, target_format="mp3"):
    if target_format == "mp3":
        codec = "libmp3lame"
    elif target_format == "wav":
        codec = "pcm_s16le"  # Codec for WAV format
    else:
        raise ValueError("Unsupported target format")
    
    subprocess.run(["ffmpeg", "-i", input_path, "-codec:a", codec, output_path])

def get_all_wav_files(input_dir):
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def process_files(wav_files, input_dir, output_dir_mp3, output_dir_wav):
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # Maintain directory structure in output
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # Create corresponding output directories
        output_dir_mp3_path = os.path.join(output_dir_mp3, relative_path)
        output_dir_wav_path = os.path.join(output_dir_wav, relative_path)
        
        os.makedirs(output_dir_mp3_path, exist_ok=True)
        os.makedirs(output_dir_wav_path, exist_ok=True)
        
        # Generate output file paths
        output_file_mp3 = os.path.join(output_dir_mp3_path, os.path.basename(input_file_path).replace(".wav", ".mp3"))
        output_file_wav = os.path.join(output_dir_wav_path, os.path.basename(input_file_path))
        
        # Perform format conversions
        convert_format(input_file_path, output_file_mp3, target_format="mp3")
        convert_format(output_file_mp3, output_file_wav, target_format="wav")

if __name__ == "__main__":
    input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/it3/test-clean"
    output_dir_mp3 = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/format-mp3/test-clean"
    output_dir_wav = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/format-wav/test-clean"
    
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    process_files(wav_files, input_dir, output_dir_mp3, output_dir_wav)
    
    print("All files processed and saved.")
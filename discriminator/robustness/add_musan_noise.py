import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random

def add_musan_noise(input_path, output_path, noise_dir, noise_level=0.01):
    # Load audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Randomly select a noise file from MUSAN directory
    noise_file = random.choice(get_all_wav_files(noise_dir))
    noise, _ = librosa.load(noise_file, sr=sr)  # Ensure noise has same sample rate as input
    
    # Trim noise if longer than audio, repeat if shorter
    if len(noise) > len(y):
        noise = noise[:len(y)]
    else:
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))[:len(y)]
    
    # Adjust noise intensity
    noise *= noise_level
    
    # Mix noise with original audio
    y_noisy = y + noise
    
    # Save noisy audio file
    sf.write(output_path, y_noisy, sr)

def get_all_wav_files(input_dir):
    # Recursively find all .wav files in directory
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def process_files(wav_files, input_dir, output_dir, noise_dir):
    # Process each file with progress bar
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # Maintain directory structure in output
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        output_file_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_file_dir, exist_ok=True)
        
        # Generate output file path
        output_file_path = os.path.join(output_file_dir, os.path.basename(input_file_path))
        
        # Add MUSAN noise and save
        add_musan_noise(input_file_path, output_file_path, noise_dir)

if __name__ == "__main__":
    input_dir = "/PATH/TO/INPUT/DIR"
    output_dir = "PATH/TO/OUTPUT/DIR"
    noise_dir = "PATH/TO/MUSAN/DIR"  
    
    print("Collecting .wav files from input directory...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")

    process_files(wav_files, input_dir, output_dir, noise_dir)
    
    print("All files processed and saved.")

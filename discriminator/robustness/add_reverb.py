import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def add_reverb(input_path, output_path, delay=0.5, decay=0.5):
    # Load audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Calculate delay in samples
    delay_samples = int(delay * sr)
    
    # Create decaying reflection
    reverb_tail = np.zeros_like(y)
    for i in range(delay_samples, len(y)):
        reverb_tail[i] = y[i - delay_samples] * decay
    
    # Add reverb to original audio
    y_reverb = y + reverb_tail
    
    # Normalize audio to prevent clipping
    y_reverb = y_reverb / np.max(np.abs(y_reverb))
    
    # Save processed audio
    sf.write(output_path, y_reverb, sr)

def get_all_wav_files(input_dir):
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def process_files(wav_files, input_dir, output_dir):

    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # Calculate relative path to maintain directory structure
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # Create corresponding output directory
        output_file_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_file_dir, exist_ok=True)
        
        # Generate output file path
        output_file_path = os.path.join(output_file_dir, os.path.basename(input_file_path))
        
        # Add reverb and save
        add_reverb(input_file_path, output_file_path)

if __name__ == "__main__":
    input_dir = "/PATH/TO/INPUT/DIR"
    output_dir = "/PATH/TO/OUTPUT/DIR"
    
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    process_files(wav_files, input_dir, output_dir)
    
    print("All files processed and saved.")
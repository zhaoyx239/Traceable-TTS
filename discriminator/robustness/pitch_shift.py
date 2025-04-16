import os
import librosa
import soundfile as sf
from tqdm import tqdm

def pitch_shift(input_path, output_path, n_steps=4):
    y, sr = librosa.load(input_path, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_path, y_shifted, sr)

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
        
        # Apply pitch shift
        pitch_shift(input_file_path, output_file_path, n_steps=4)

if __name__ == "__main__":
    input_dir = "/PATH/TO/INPUT/DIR"
    output_dir = "/PATH/TO/OUTPUT/DIR"

    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    process_files(wav_files, input_dir, output_dir)
    
    print("All files processed and saved.")
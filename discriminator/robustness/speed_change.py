import os
import librosa
import soundfile as sf
from tqdm import tqdm

def change_speed(input_path, output_path, speed_factor):
    # Load audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Change speed using librosa's time_stretch function
    y_stretched = librosa.effects.time_stretch(y, rate = speed_factor)
    
    # Save speed-changed audio file
    sf.write(output_path, y_stretched, sr)

def get_all_wav_files(input_dir):
    # Recursively find all .wav files in directory
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def process_files(wav_files, input_dir, output_dir_1_2, output_dir_0_8):
    # Process each file with progress bar
    for input_file_path in tqdm(wav_files, desc="Processing files"):
        # Maintain directory structure in output
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        
        # Create corresponding output directories
        output_dir_1_2_path = os.path.join(output_dir_1_2, relative_path)
        output_dir_0_8_path = os.path.join(output_dir_0_8, relative_path)
        
        os.makedirs(output_dir_1_2_path, exist_ok=True)
        os.makedirs(output_dir_0_8_path, exist_ok=True)
        
        # Generate output file paths
        output_file_1_2 = os.path.join(output_dir_1_2_path, os.path.basename(input_file_path))
        output_file_0_8 = os.path.join(output_dir_0_8_path, os.path.basename(input_file_path))
        
        # Process with 1.2x speed
        change_speed(input_file_path, output_file_1_2, 1.2)
        
        # Process with 0.8x speed
        change_speed(input_file_path, output_file_0_8, 0.8)

if __name__ == "__main__":
    # Input directory
    input_dir = "/PATH/TO/INPUT/DIR"
    
    # Output directories
    output_dir_1_2 = "/PATH/TO/OUTPUT/DIR/1.2"
    output_dir_0_8 = "/PATH/TO/OUTPUT/DIR/0.8"
    
    # Get all .wav file paths
    print("Collecting .wav files...")
    wav_files = get_all_wav_files(input_dir)
    print(f"Found {len(wav_files)} .wav files.")
    
    # Process all files
    process_files(wav_files, input_dir, output_dir_1_2, output_dir_0_8)
    
    print("All files processed and saved.")
import os
import torchaudio
from torchaudio.transforms import Resample
from datetime import datetime
import threading
from tqdm import tqdm

# Input and output paths
input_dir = "/PATH/TO/INPUT/DIR"
output_base_dir = "/PATH/TO/OUTPUT/DIR"  

# Limit for concurrent threads
thread_limit = 16
threads = []

def resample_audio(input_path, output_path, orig_freq=24000, new_freq=8000):
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Perform resampling
        resampler = Resample(orig_freq=sample_rate, new_freq=new_freq)
        resampled_waveform = resampler(waveform)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save resampled audio
        torchaudio.save(output_path, resampled_waveform, new_freq)
        tqdm.write(f"[{datetime.now()}] Resampling complete: {input_path} -> {output_path}")
    except Exception as e:
        tqdm.write(f"Error: Problem processing file {input_path}: {e}")

# Collect all files to process
file_pairs = []  # Store (input_path, output_path)
for root, _, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".wav"):  # Process only .wav files
            input_path = os.path.join(root, file_name)
            
            # Construct output path while maintaining directory structure
            relative_path = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            output_path = os.path.join(output_dir, file_name)

            # Skip if output already exists
            if os.path.exists(output_path):
                continue
            file_pairs.append((input_path, output_path))

# Show progress with tqdm
with tqdm(total=len(file_pairs), desc="Resampling progress") as pbar:
    for input_path, output_path in file_pairs:
        # Create thread for resampling
        thread = threading.Thread(target=resample_audio, args=(input_path, output_path))
        threads.append(thread)
        thread.start()

        # Control number of concurrent threads
        if len(threads) >= thread_limit:
            for t in threads:
                t.join()  # Wait for current threads to finish
                pbar.update(1)  # Update progress bar
            threads.clear()  # Clear thread list

    # Wait for remaining threads
    for t in threads:
        t.join()
        pbar.update(1)  # Update progress bar
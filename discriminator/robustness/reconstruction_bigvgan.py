import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm
from BigVGAN.bigvgan import BigVGAN
from meldataset import get_mel_spectrogram

# Define paths
input_dir = "/PATH/TO/INPUT/DIR"
output_dir_bigvgan = "/PATH/TO/OUTPUT/DIR"
bigvgan_model_path = "/PATH/TO/BIGVGAN/MODEL"

# Set GPU configuration
gpu_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BigVGAN model from local path
def load_bigvgan_from_local(local_path, device):
    model = BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model

# Main function
if __name__ == "__main__":
    # Load BigVGAN model
    print("Loading BigVGAN model...")
    bigvgan_model = load_bigvgan_from_local(bigvgan_model_path, device)
    
    # Wrap model with DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        bigvgan_model = DataParallel(bigvgan_model, device_ids=gpu_ids)
    
    # Get all .wav file paths
    print("Scanning directory for audio files...")
    file_paths = [str(p) for p in Path(input_dir).rglob("*.wav")]
    print(f"Found {len(file_paths)} audio files.")
    
    # Process files with progress bar
    for file_path in tqdm(file_paths, desc="Processing audio files", unit="file"):
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(device)
        
        # Extract Mel spectrogram and generate audio
        with torch.no_grad():
            # Extract Mel spectrogram for BigVGAN
            mel_bigvgan = get_mel_spectrogram(waveform, bigvgan_model.h)
            
            # Generate audio using BigVGAN
            audio_bigvgan = bigvgan_model(mel_bigvgan).squeeze(0).cpu()
        
        # Build output path
        relative_path = os.path.relpath(file_path, input_dir)
        output_path_bigvgan = os.path.join(output_dir_bigvgan, relative_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path_bigvgan), exist_ok=True)
        
        # Save generated audio
        sf.write(output_path_bigvgan, audio_bigvgan.numpy().T, sr)

    

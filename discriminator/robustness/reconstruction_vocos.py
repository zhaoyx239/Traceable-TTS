import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from vocos import Vocos

# Define paths
input_dir = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/waterMarked/groundtruth/test-clean"
output_dir_vocos = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/waterMarked/vocos/test-clean"
vocos_model_path = "/hpc_stor03/sjtu_home/yuxiang.zhao/F5-TTS/vocos-mel-24khz"

# Set GPU configuration
gpu_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Vocos model from local path
def load_vocos_from_local(local_path, device):
    config_path = os.path.join(local_path, "config.yaml")
    model_path = os.path.join(local_path, "pytorch_model.bin")
    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    from vocos.feature_extractors import EncodecFeatures
    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder

# Main function
if __name__ == "__main__":
    # Load Vocos model
    print("Loading Vocos model...")
    vocos_model = load_vocos_from_local(vocos_model_path, device)
    
    # Get all .wav file paths
    print("Scanning directory for audio files...")
    file_paths = [str(p) for p in Path(input_dir).rglob("*.wav")]
    print(f"Found {len(file_paths)} audio files.")
    
    # Process files with progress bar
    for file_path in tqdm(file_paths, desc="Processing audio files", unit="file"):
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(device)
        
        # Generate audio using Vocos
        with torch.no_grad():
            audio_vocos = vocos_model(waveform).cpu()
        
        # Build output path
        relative_path = os.path.relpath(file_path, input_dir)
        output_path_vocos = os.path.join(output_dir_vocos, relative_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path_vocos), exist_ok=True)
        
        # Save generated audio
        sf.write(output_path_vocos, audio_vocos.numpy().T, sr)
    

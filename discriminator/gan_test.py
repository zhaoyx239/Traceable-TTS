import torch
import os
from transformers import Wav2Vec2Processor
from models.wav2vec_classifier import Wav2VecClassifier
from torchaudio.transforms import Resample


# Load model with module
def load_model_with_module(model, model_path, device):
    if os.path.exists(model_path):
        # Load model parameters
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model

def check(audio_tensor):
    # Load Wav2Vec2 processor
    processor = Wav2Vec2Processor.from_pretrained("/PATH/TO/wav2vec2")
    
    # Select device
    device = audio_tensor.device
    
    # Initialize model
    model = Wav2VecClassifier()
    
    # Load trained model
    model = load_model_with_module(model, '/PATH/TO/model_best.pth', device)
    model.to(device)
    model.eval()
    
    watermark_scores = []
    for audio in audio_tensor:
        audio = audio.to(torch.float32)
        # Resample to 16kHz
        resampler = Resample(24000, new_freq=16000)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
        
        # Process audio with Wav2Vec2 processor
        inputs = processor(
                audio.squeeze().cpu().numpy(),  # Audio signal (must be numpy array)
                sampling_rate=16000,            # Audio sample rate
                return_tensors="pt"             # Return PyTorch tensors
            )
        input_values = inputs.input_values.to(device)  # Model input features
        
        # Perform inference
        with torch.no_grad():
            output = model(input_values).squeeze().item()  # Get detection score
        watermark_scores.append(output)
    
    return watermark_scores
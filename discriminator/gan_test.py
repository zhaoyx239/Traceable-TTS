import torch
import os
from transformers import Wav2Vec2Processor
from models.wav2vec_classifier import Wav2VecClassifier
from torchaudio.transforms import Resample

def load_model_with_module(model, model_path, device):
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model

def check(audio_tensor):
    processor = Wav2Vec2Processor.from_pretrained("/PATH/TO/wav2vec2")
    device = audio_tensor.device
    model = Wav2VecClassifier()

    model = load_model_with_module(model, '/PATH/TO/model_best.pth', device)
    model.to(device)
    model.eval()
    
    watermark_scores = []
    for audio in audio_tensor:
        audio = audio.to(torch.float32)
        resampler = Resample(24000, new_freq=16000)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)

        inputs = processor(
                audio.squeeze().cpu().numpy(), 
                sampling_rate=16000,           
                return_tensors="pt"            
            )
        input_values = inputs.input_values.to(device)  

        with torch.no_grad():
            output = model(input_values).squeeze().item() 
        watermark_scores.append(output)
    
    return watermark_scores
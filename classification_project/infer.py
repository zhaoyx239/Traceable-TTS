# infer.py
import torch
from transformers import Wav2Vec2Processor
from utils.preprocess import load_audio
from models.wav2vec_classifier import Wav2VecClassifier

def predict(file_path, model_path='wav2vec_audio_classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和处理器
    model = Wav2VecClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    waveform = load_audio(file_path)
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    with torch.no_grad():
        output = model(input_values).item()
    
    if output > 0.5:
        print("Prediction: Real Audio")
    else:
        print("Prediction: Synthetic Audio")

# 使用示例
predict('path/to/test_audio.wav')

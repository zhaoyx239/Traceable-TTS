import torch
import os
from transformers import Wav2Vec2Processor
from models.wav2vec_classifier import Wav2VecClassifier
from torchaudio.transforms import Resample


# 加载模型
def load_model_with_module(model, model_path, device):
    if os.path.exists(model_path):
        # 加载模型参数
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model

def check(audio_tensor):
    # 加载 processor
    processor = Wav2Vec2Processor.from_pretrained("/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base")
    
    # 选择设备
    device = audio_tensor.device
    
    # 初始化模型
    model = Wav2VecClassifier()
    
    # 加载模型
    model = load_model_with_module(model, '/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/model_best.pth', device)
    model.to(device)
    model.eval()
    watermark_scores = []
    for audio in audio_tensor:
        audio = audio.to(torch.float32)
        # 重采样为16kHz
        resampler = Resample(24000, new_freq=16000)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
        # 检测水印
        # 使用 processor 处理音频
        inputs = processor(
                audio.squeeze().cpu().numpy(),  # 音频信号（需要是 numpy 数组）
                sampling_rate=16000,                  # 音频采样率
                return_tensors="pt"                   # 返回 PyTorch 张量
            )
        input_values = inputs.input_values.to(device)  # 模型输入的特征
        # 推理
        with torch.no_grad():
            output = model(input_values).squeeze().item()  # 获取检测得分  
        watermark_scores.append(output)
    

    return watermark_scores
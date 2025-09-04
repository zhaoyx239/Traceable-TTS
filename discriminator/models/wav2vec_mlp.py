import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecClassifier(nn.Module):
    def __init__(self,wav2vec2_version="base"):
        super(Wav2VecClassifier, self).__init__()
        if wav2vec2_version=="xlsr300m":
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-xls-r-300m"
            )
        elif wav2vec2_version=="base":
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base"
            )
        else:
            raise ValueError(f"Unsupported wav2vec2 version: {wav2vec2_version}")
        self.feature_dim = self.wav2vec.config.hidden_size  # Wav2Vec2 输出特征维度
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256), 
            nn.ReLU(),                       
            nn.Dropout(0.5),                 
            nn.Linear(256, 128),             
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) 
        )

    def forward(self, input_audio):
        with torch.no_grad():
            features = self.wav2vec(input_audio).last_hidden_state  # (batch, seq_len, feature_dim)

        # 调整维度以适配 Conv1d
        features = features.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        # 填充到最小时长
        if(features.shape[-1]<128):
            features = torch.nn.functional.pad(features,(0,128-features.shape[-1]))
        x = torch.mean(features,dim = 2)
        x = self.mlp_head(x)
        return x

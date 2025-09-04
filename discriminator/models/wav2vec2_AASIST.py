import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor to help the model capture sequence order.
    """
    def __init__(self, d_model, max_len=7000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class AttentionModule(nn.Module):
    """
    Implements a self-attention mechanism to capture dependencies between frames.
    """
    def __init__(self, input_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        return attn_output

class RawBoost(nn.Module):
    """
    RawBoost preprocessing module as described in the AASIST paper.
    """
    def __init__(self, channel_in):
        super(RawBoost, self).__init__()
        self.conv = nn.Conv1d(channel_in, 16, kernel_size=11, stride=1, padding=5)
        self.bn = nn.BatchNorm1d(16)

    def forward(self, x):
        # 打印输入数据的形状，检查数据是否正确传递
        # print(f"Input to RawBoost: {x.shape}")
        x = x.unsqueeze(1)
        # 执行卷积，输出 shape 为 [batch_size, 16, seq_length]
        x = self.conv(x)
        # print(f"After Conv1d: {x.shape}")
        
        # 执行 BatchNorm 归一化
        x = self.bn(x)
        # print(f"After BatchNorm: {x.shape}")
        
        # ReLU 激活
        x = F.relu(x)
        return x

class Wav2VecFrontEnd(nn.Module):
    """
    Frontend using Wav2Vec to extract high-level speech features.
    """
    def __init__(self, wav2vec_model_name="/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base"):
        super(Wav2VecFrontEnd, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)

    def forward(self, x, attention_mask=None):
        # Wav2Vec expects input as [batch_size, sequence_length]
        # If input is [batch_size, channels, sequence_length], squeeze the channel dimension
        if x.dim() == 3:
            x = x.squeeze(1)
        with torch.no_grad():
            outputs = self.wav2vec(input_values=x, attention_mask=attention_mask)
        # Extract last hidden state
        return outputs.last_hidden_state

class AASISTWithWav2Vec(nn.Module):
    def __init__(self, input_channels, num_classes, wav2vec_model_name="/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base"):
        super(AASISTWithWav2Vec, self).__init__()
        self.rawboost = RawBoost(input_channels)
        self.wav2vec_frontend = Wav2VecFrontEnd(wav2vec_model_name)
        self.positional_encoding = PositionalEncoding(d_model=768)  # Wav2Vec2 output dim is 768
        self.attention_module = AttentionModule(input_dim=768, num_heads=4)
        self.fc = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Apply RawBoost preprocessing
        x = self.rawboost(x)  # Output: [batch_size, 16, seq_length]
        
        # Step 2: Reduce the channel dimension for Wav2Vec compatibility
        x = x.mean(dim=1)  # Collapse channel dimension: [batch_size, seq_length]

        # Step 3: Extract high-level features using Wav2Vec frontend
        x = self.wav2vec_frontend(x)  # Output: [batch_size, seq_length, 768]

        # Step 4: Add positional encoding
        x = self.positional_encoding(x)

        # Step 5: Self-attention to capture temporal dependencies
        x = self.attention_module(x)

        # Step 6: Global average pooling
        x = x.mean(dim=1)  # Average over the temporal dimension

        # Step 7: Fully connected layer for classification
        x = self.fc(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x


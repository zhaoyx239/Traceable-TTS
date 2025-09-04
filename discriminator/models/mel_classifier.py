import torch
import torch.nn as nn
from torchvision.models import resnet34
import torchaudio
import torchaudio.functional as F


class MelResNetClassifier(nn.Module):
    def __init__(self, input_type='audio', num_classes=1, sr=24000,  # 默认24kHz
                 n_fft=1024, hop_length=256, n_mels=128):           # 24kHz推荐参数
        """
        初始化模型
        :param input_type: 输入数据类型，'audio' 或 'mel'
        :param num_classes: 输出类别数
        :param sr: 采样率（仅当 input_type='audio' 时使用）
        :param n_fft: FFT窗口大小（仅当 input_type='audio' 时使用）
        :param hop_length: 帧移（仅当 input_type='audio' 时使用）
        :param n_mels: 梅尔频带数（仅当 input_type='audio' 时使用）
        """
        super().__init__()
        self.input_type = input_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # 定义音频到频谱的转换模块

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )

        # 修改ResNet结构
        self.resnet = resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，可以是音频或频谱
        :return: 模型输出
        """
        # 如果输入是音频，转换为频谱
        if self.input_type == 'audio':
            if isinstance(x, str):  # 如果输入是文件路径
                x, _ = torchaudio.load(x)
            x = self.mel_transform(x)  # 转换为梅尔频谱
            x = torch.log10(x + 1e-9)  # 对数变换

        # 如果输入是频谱，直接使用
        elif self.input_type == 'mel':
            pass

        # 确保输入形状为 (batch_size, 1, n_mels, time)
        if len(x.shape) == 2:  # 如果是 (n_mels, time)，增加 batch 和 channel 维度
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:  # 如果是 (batch_size, n_mels, time)，增加 channel 维度
            x = x.unsqueeze(1)

        # 通过模型
        x = self.resnet(x)
        return torch.sigmoid(x).squeeze()  # 去掉多余的维度
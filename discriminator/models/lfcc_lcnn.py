import torch
import torch.nn as nn
from torchaudio.transforms import LFCC

class LFCC_LCNN(nn.Module):
    def __init__(self):
        super(LFCC_LCNN, self).__init__()
        # LFCC特征提取器
        self.lfcc = LFCC(
            sample_rate=24000
        )

        # 定义LCNN结构，使用conv2d
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batch_norm6 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.batch_norm7 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.7)

        # 全局池化
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_audio):
        # 提取LFCC特征
        lfcc_features = self.lfcc(input_audio)  # (batch, 1, time, freq)
        # print(lfcc_features.shape)
        # 调整维度顺序：LFCC 输出为 (batch, time, freq)，需要调整为 (batch, 1, freq, time)
        # lfcc_features = lfcc_features.permute(0, 1, 3, 2)  # 转换为 (batch, 1, freq, time)

        # 通过卷积层、BatchNorm 和激活函数
        x = self.conv1(lfcc_features)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.max_pool2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = self.max_pool3(x)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = self.batch_norm5(x)
        x = torch.relu(x)
        x = self.conv7(x)
        x = self.max_pool4(x)
        x = self.batch_norm6(x)
        x = torch.relu(x)
        x = self.conv8(x)
        x = self.batch_norm7(x)
        x = torch.relu(x)
        x = self.conv9(x)
        x = self.max_pool5(x)
        x = self.dropout(x)

        # 全局池化
        x = self.adaptive_avg_pool(x)  # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 64)

        # 全连接层和分类
        x = self.linear(x)  # (batch, 1)
        return self.sigmoid(x)

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecClassifier(nn.Module):
    def __init__(self):
        super(Wav2VecClassifier, self).__init__()

        # Load pretrained Wav2Vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "/PATH/TO/wav2vec2-base"
        )
        self.feature_dim = self.wav2vec.config.hidden_size  # Wav2Vec2 output feature dimension

        # Define LCNN convolutional blocks
        self.conv1 = nn.Conv1d(self.feature_dim, 64, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 96, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm1d(96)
        
        self.conv4 = nn.Conv1d(96, 96, kernel_size=1, stride=1)
        self.batch_norm3 = nn.BatchNorm1d(96)

        self.conv5 = nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm4 = nn.BatchNorm1d(128)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=1, stride=1)
        self.batch_norm5 = nn.BatchNorm1d(128)

        self.conv7 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm6 = nn.BatchNorm1d(64)

        self.conv8 = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.batch_norm7 = nn.BatchNorm1d(64)

        self.conv9 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.7)

        # Global pooling
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_audio):
        # Extract features from Wav2Vec2
        features = self.wav2vec(input_audio).last_hidden_state  # (batch, seq_len, feature_dim)

        # Adjust dimensions for Conv1d
        features = features.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        # Pad to minimum length
        if features.shape[-1] < 128:
            features = torch.nn.functional.pad(features, (0, 128 - features.shape[-1]))

        # Through LCNN convolutional layer, BatchNorm and activation function
        x = self.conv1(features)  # (batch, 64, seq_len)
        x = self.max_pool1(x)     # (batch, 64, seq_len//2)
        x = self.conv2(x)         # (batch, 64, seq_len//2)
        x = self.batch_norm1(x)
        x = torch.relu(x)

        x = self.conv3(x)         # (batch, 96, seq_len//2)
        x = self.max_pool2(x)     # (batch, 96, seq_len//4)
        x = self.batch_norm2(x)
        x = torch.relu(x)

        x = self.conv4(x)         # (batch, 96, seq_len//4)
        x = self.batch_norm3(x)
        x = torch.relu(x)

        x = self.conv5(x)         # (batch, 128, seq_len//4)
        x = self.max_pool3(x)     # (batch, 128, seq_len//8)
        x = self.batch_norm4(x)
        x = torch.relu(x)

        x = self.conv6(x)         # (batch, 128, seq_len//8)
        x = self.batch_norm5(x)
        x = torch.relu(x)

        x = self.conv7(x)         # (batch, 64, seq_len//8)
        x = self.max_pool4(x)     # (batch, 64, seq_len//16)
        x = self.batch_norm6(x)
        x = torch.relu(x)

        x = self.conv8(x)         # (batch, 64, seq_len//16)
        x = self.batch_norm7(x)
        x = torch.relu(x)

        x = self.conv9(x)         # (batch, 64, seq_len//16)
        x = self.max_pool5(x)     # (batch, 64, seq_len//32)
        x = self.dropout(x)

        # Global pooling
        x = self.adaptive_avg_pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)             # (batch, 64)

        # Fully connected layer and classification
        x = self.linear(x)  # (batch, 1)
        return self.sigmoid(x)

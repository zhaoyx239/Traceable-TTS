import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import Wav2Vec2Processor
from utils.preprocess import load_audio
from models.wav2vec_classifier import Wav2VecClassifier
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    input_values = pad_sequence(input_values, batch_first=True)
    return input_values, labels

# 数据集类
class AudioDataset(Dataset):
    def __init__(self, data_dir, label, processor):
        self.files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(data_dir) 
            for f in files if f.endswith(".wav")
        ]
        self.label = label
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform = load_audio(file_path)
        input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        return input_values.squeeze(), torch.tensor(self.label, dtype=torch.float32)

# 数据集划分函数
def create_data_loaders(train_data_real, val_data_real, test_data_real, train_data_synthetic, val_data_synthetic, test_data_synthetic, processor, batch_size=8):
    train_real = AudioDataset(train_data_real, 1, processor)
    val_real = AudioDataset(val_data_real, 1, processor)
    test_real = AudioDataset(test_data_real, 1, processor)

    train_synthetic = AudioDataset(train_data_synthetic, 0, processor)
    val_synthetic = AudioDataset(val_data_synthetic, 0, processor)
    test_synthetic = AudioDataset(test_data_synthetic, 0, processor)

    # 合并真实和合成数据集
    train_dataset = train_real + train_synthetic
    val_dataset = val_real + val_synthetic
    test_dataset = test_real + test_synthetic

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_values, labels in data_loader:
            input_values, labels = input_values.to(device), labels.to(device)
            outputs = model(input_values)
            predicted = (outputs > 0.5).float()
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def main():
    # 初始化 processor
    processor = Wav2Vec2Processor.from_pretrained("/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base")

    # 设置真实和合成数据路径
    real_train_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/train-clean-100"
    real_val_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/dev-clean"
    real_test_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/test-clean"

    synthetic_train_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts/train-clean-100"
    synthetic_val_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts/dev-clean"
    synthetic_test_data = "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts/test-clean"

    batch_size = 8

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        real_train_data, real_val_data, real_test_data,
        synthetic_train_data, synthetic_val_data, synthetic_test_data,
        processor,
        batch_size=batch_size
    )

    # 初始化模型
    model = Wav2VecClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 指定 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 DataParallel 包装模型
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    # 训练模型
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for input_values, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            input_values, labels = input_values.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 计算验证集上的损失
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_values, labels in tqdm(val_loader, desc="Validation"):
                input_values, labels = input_values.to(device), labels.to(device)
                outputs = model(input_values)
                val_loss = criterion(outputs.squeeze(), labels)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        

        # 保存模型
        if (epoch+1) % 1 ==0:
            torch.save(model.state_dict(), f'wav2vec_audio_classifier_with_grad_{epoch+1}.pth')
            # 计算训练、验证和测试集准确率
            train_accuracy = evaluate(model, train_loader, device)
            val_accuracy = evaluate(model, val_loader, device)
            test_accuracy = evaluate(model, test_loader, device)
            print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()

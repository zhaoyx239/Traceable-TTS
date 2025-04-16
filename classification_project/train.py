import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 使用 0, 1, 2, 3 号 GPU
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from utils.preprocess import load_audio
from models.wav2vec_classifier import Wav2VecClassifier
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# 数据整理函数
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

# 创建训练数据加载器
def create_train_loader(real_data_dirs, synthetic_data_dirs, processor, batch_size=32):  # 增加 batch_size
    # 真实数据标签为 0，合成数据标签为 1
    train_real_datasets = [AudioDataset(real_data_dir, 0, processor) for real_data_dir in real_data_dirs]
    train_synthetic_datasets = [AudioDataset(synthetic_data_dir, 1, processor) for synthetic_data_dir in synthetic_data_dirs]

    # 合并数据集
    train_dataset = torch.utils.data.ConcatDataset(train_real_datasets + train_synthetic_datasets)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader

def main():
    # 初始化 processor
    processor = Wav2Vec2Processor.from_pretrained("/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base")

    # 设置真实数据和合成数据路径
    real_train_data_dirs = [
        "/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/dev-clean",
        # "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts/dev-clean",
        # "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice/dev-clean"
        
    ]
    synthetic_train_data_dirs = [
        "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer-simple/f5tts-mask/dev-clean"
    ]

    batch_size = 32  # 增加 batch_size

    # 创建训练数据加载器
    train_loader = create_train_loader(real_train_data_dirs, synthetic_train_data_dirs, processor, batch_size=batch_size)

    # 初始化模型
    model = Wav2VecClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-4)

    # 指定 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    model.to(device)  # 将模型移动到设备

    # 检查是否存在 model_last.pth 文件
    model_path = "model_last.pth"
    best_model_path = "model_best.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} to continue training...")
        model.module.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型权重
    else:
        print("No existing model found. Training from scratch...")

    # 训练模型
    num_epochs = 20
    best_train_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for input_values, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            input_values, labels = input_values.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        # 如果当前 epoch 的训练损失是最小的，则保存模型
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.module.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with train loss: {best_train_loss:.4f}")
    # 训练结束后保存模型
    torch.save(model.module.state_dict(), model_path)  # 保存模型时使用 .module
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
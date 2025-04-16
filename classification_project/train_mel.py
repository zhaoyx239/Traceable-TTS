import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.mel_classifier import MelResNetClassifier

# ----------------- 修复后的数据整理函数 -----------------
def collate_fn(batch):
    waveforms = []
    labels = []
    for wave, label in batch:
        if label != -1:  # 过滤无效样本
            waveforms.append(wave)
            labels.append(label)
    
    if not waveforms:
        return torch.zeros(0), torch.zeros(0)
    
    # 动态计算最大长度（限制在10秒）
    max_len = min(max(w.size(0) for w in waveforms), 24000 * 10)
    
    # 使用 torch.nn.functional.pad 进行填充
    padded = []
    for w in waveforms:
        if w.size(0) < max_len:
            # 如果音频长度小于 max_len，填充到 max_len
            pad_size = max_len - w.size(0)
            padded_wave = torch.nn.functional.pad(w, (0, pad_size), mode='constant', value=0)
        else:
            # 如果音频长度大于 max_len，裁剪到 max_len
            padded_wave = w[:max_len]
        padded.append(padded_wave)
    
    return torch.stack(padded), torch.stack(labels)

# ----------------- 简化数据集类 -----------------
class SimpleAudioDataset(Dataset):
    def __init__(self, data_dirs, label):
        self.files = []
        for d in data_dirs:
            for root, _, files in os.walk(d):
                self.files.extend([os.path.join(root,f) for f in files if f.endswith(('.wav','.flac'))])
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            waveform, _ = torchaudio.load(self.files[idx])  # 加载音频
            waveform = waveform.mean(dim=0)  # 转单声道
            return waveform, torch.tensor(self.label, dtype=torch.float32)
        except:
            return torch.zeros(1), -1  # 错误样本标记

# ----------------- 简化训练流程 -----------------
def main():
    # 初始化模型
    model = MelResNetClassifier(input_type='audio')  # 输入类型为音频
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    
    # 数据路径
    real_dirs = ["/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/dev-clean",
        "/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice/dev-clean"]
    fake_dirs = ["/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer_audio/f5tts-mask/dev-clean"]
    
    # 创建数据集
    train_set = torch.utils.data.ConcatDataset([
        SimpleAudioDataset(real_dirs, 0),
        SimpleAudioDataset(fake_dirs, 1)
    ])
    
    # 数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 训练配置
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    
    # 恢复训练逻辑简化
    best_loss = float('inf')
    if os.path.exists("model_last.pth"):
        model.module.load_state_dict(torch.load("model_last.pth"))
        print("Loaded previous checkpoint")
    
    # 固定训练轮次
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for waves, labels in pbar:
            if waves.nelement() == 0: continue  # 跳过空批次
            
            waves = waves.to(device)
            labels = labels.to(device)
            # print(f"waves shape:{waves.shape}")
            # print(f"label shape:{labels.shape}")
            optimizer.zero_grad()
            outputs = model(waves)  # 传入音频张量
            # print(f"outputs shape:{outputs.shape}")
            # print(f"label shape:{labels.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # 保存逻辑简化
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.module.state_dict(), "model_best.pth")
            print("Saved new best model")
    
    torch.save(model.module.state_dict(), "model_last.pth")

if __name__ == "__main__":
    main()
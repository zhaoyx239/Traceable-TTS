import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.mel_classifier import MelResNetClassifier

def collate_fn(batch):
    waveforms = []
    labels = []
    for wave, label in batch:
        if label != -1: 
            waveforms.append(wave)
            labels.append(label)
    
    if not waveforms:
        return torch.zeros(0), torch.zeros(0)
    
    max_len = min(max(w.size(0) for w in waveforms), 24000 * 10)

    padded = []
    for w in waveforms:
        if w.size(0) < max_len:
            pad_size = max_len - w.size(0)
            padded_wave = torch.nn.functional.pad(w, (0, pad_size), mode='constant', value=0)
        else:
            padded_wave = w[:max_len]
        padded.append(padded_wave)
    
    return torch.stack(padded), torch.stack(labels)

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
            waveform, _ = torchaudio.load(self.files[idx]) 
            waveform = waveform.mean(dim=0) 
            return waveform, torch.tensor(self.label, dtype=torch.float32)
        except:
            return torch.zeros(1), -1 

def main():
    model = MelResNetClassifier(input_type='audio') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    real_dirs = ["/PATH/TO/DATA"]
    fake_dirs = ["/PATH/TO/DATA"]

    train_set = torch.utils.data.ConcatDataset([
        SimpleAudioDataset(real_dirs, 0),
        SimpleAudioDataset(fake_dirs, 1)
    ])

    train_loader = DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    best_loss = float('inf')
    if os.path.exists("model_last.pth"):
        model.module.load_state_dict(torch.load("model_last.pth"))
        print("Loaded previous checkpoint")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for waves, labels in pbar:
            if waves.nelement() == 0: continue 
            
            waves = waves.to(device)
            labels = labels.to(device)
            # print(f"waves shape:{waves.shape}")
            # print(f"label shape:{labels.shape}")
            optimizer.zero_grad()
            outputs = model(waves) 
            # print(f"outputs shape:{outputs.shape}")
            # print(f"label shape:{labels.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.module.state_dict(), "model_best.pth")
    
    torch.save(model.module.state_dict(), "model_last.pth")

if __name__ == "__main__":
    main()
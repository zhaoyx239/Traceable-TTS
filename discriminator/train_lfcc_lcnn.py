import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from models.lfcc_lcnn import LFCC_LCNN  
from utils.preprocess import load_audio
from tqdm import tqdm
import torch.nn.functional as F
import wandb  

def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    max_length = max([x.shape[1] for x in input_values]) 
    padded_input_values = []
    for x in input_values:
        padding_length = max_length - x.shape[1]  
        if padding_length > 0:
            padded_x = F.pad(x, (0, padding_length))  
        else:
            padded_x = x
        padded_input_values.append(padded_x)

    input_values = torch.stack(padded_input_values)
    
    return input_values, labels

class AudioDataset(Dataset):
    def __init__(self, data_dir, label):
        self.files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(data_dir) 
            for f in files if f.endswith(".wav")
        ]
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform = load_audio(file_path)
        return waveform, torch.tensor(self.label, dtype=torch.float32)

def create_data_loaders(train_data_real, val_data_real, test_data_real, train_data_synthetic, val_data_synthetic, test_data_synthetic, batch_size=8):
    train_real = AudioDataset(train_data_real, 1)
    val_real = AudioDataset(val_data_real, 1)
    test_real = AudioDataset(test_data_real, 1)

    train_synthetic = AudioDataset(train_data_synthetic, 0)
    val_synthetic = AudioDataset(val_data_synthetic, 0)
    test_synthetic = AudioDataset(test_data_synthetic, 0)

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
    wandb.init(project="audio-deepfake-detection", name="lfcc_lcnn_vocos")

    real_train_data = "/PATH/TO/DATA"
    real_val_data = "/PATH/TO/DATA"
    real_test_data = "/PATH/TO/DATA"

    synthetic_train_data = "/PATH/TO/DATA"
    synthetic_val_data = "/PATH/TO/DATA"
    synthetic_test_data = "/PATH/TO/DATA"

    batch_size = 64

    train_loader, val_loader, test_loader = create_data_loaders(
        real_train_data, real_val_data, real_test_data,
        synthetic_train_data, synthetic_val_data, synthetic_test_data,
        batch_size=batch_size
    )

    model = LFCC_LCNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    wandb.watch(model, criterion, log="all", log_freq=10)

    best_test_accuracy = 0.0

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
        
        train_accuracy = evaluate(model, train_loader, device)
        val_accuracy = evaluate(model, val_loader, device)
        test_accuracy = evaluate(model, test_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        })

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Test Acc: {test_accuracy:.4f}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            print(f"New best model saved with Test Accuracy: {best_test_accuracy:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
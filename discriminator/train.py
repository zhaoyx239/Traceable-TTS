import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from utils.preprocess import load_audio
from models.wav2vec_classifier import Wav2VecClassifier
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# Data collation function
def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    input_values = pad_sequence(input_values, batch_first=True)
    return input_values, labels

# Dataset class for audio files
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

# Create training data loader
def create_train_loader(real_data_dirs, synthetic_data_dirs, processor, batch_size=32):
    # Real data labeled as 0, synthetic data as 1
    train_real_datasets = [AudioDataset(real_data_dir, 0, processor) for real_data_dir in real_data_dirs]
    train_synthetic_datasets = [AudioDataset(synthetic_data_dir, 1, processor) for synthetic_data_dir in synthetic_data_dirs]

    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(train_real_datasets + train_synthetic_datasets)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader

def main():
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("/PATH/TO/wav2vec2-base")

    # Set paths for real and synthetic data
    real_train_data_dirs = [
        "/PATH/TO/DATA"
    ]
    synthetic_train_data_dirs = [
        "/PATH/TO/DATA"
    ]

    batch_size = 32

    # Create training data loader
    train_loader = create_train_loader(real_train_data_dirs, synthetic_train_data_dirs, processor, batch_size=batch_size)

    # Initialize model
    model = Wav2VecClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-4)

    # Set up GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    # Check for existing model
    model_path = "model_last.pth"
    best_model_path = "model_best.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} to continue training...")
        model.module.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No existing model found. Training from scratch...")

    # Training loop
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
        
        # Save best model
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.module.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with train loss: {best_train_loss:.4f}")
    
    # Save final model
    torch.save(model.module.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
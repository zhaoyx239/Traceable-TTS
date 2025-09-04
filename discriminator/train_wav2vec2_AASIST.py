import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import Wav2Vec2Processor
from utils.preprocess import load_audio
from models.wav2vec2_AASIST import AASISTWithWav2Vec 
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import csv


def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    input_values = pad_sequence(input_values, batch_first=True)
    return input_values, labels


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


def create_data_loaders(train_data_real, val_data_real, test_data_real, train_data_synthetic, val_data_synthetic, test_data_synthetic, processor, batch_size=8):
    train_real = AudioDataset(train_data_real, 1, processor)
    val_real = AudioDataset(val_data_real, 1, processor)
    test_real = AudioDataset(test_data_real, 1, processor)

    train_synthetic = AudioDataset(train_data_synthetic, 0, processor)
    val_synthetic = AudioDataset(val_data_synthetic, 0, processor)
    test_synthetic = AudioDataset(test_data_synthetic, 0, processor)

    train_dataset = train_real + train_synthetic
    val_dataset = val_real + val_synthetic
    test_dataset = test_real + test_synthetic

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_values, labels in tqdm(data_loader, desc="Accuracy"):
            input_values, labels = input_values.to(device), labels.to(device)
            outputs = model(input_values)
            predicted = (outputs > 0.5).float()
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


def main():
    processor = Wav2Vec2Processor.from_pretrained("/PATH/TO/wav2vec2-base")

    real_train_data = "/PATH/TO/DATA"
    real_val_data = "/PATH/TO/DATA"
    real_test_data = "/PATH/TO/DATA"

    synthetic_train_data = "/PATH/TO/DATA"
    synthetic_val_data = "/PATH/TO/DATA"
    synthetic_test_data = "/PATH/TO/DATA"

    batch_size = 16

    train_loader, val_loader, test_loader = create_data_loaders(
        real_train_data, real_val_data, real_test_data,
        synthetic_train_data, synthetic_val_data, synthetic_test_data,
        processor,
        batch_size=batch_size
    )

    model = AASISTWithWav2Vec(input_channels=1, num_classes=1, wav2vec_model_name="/PATH/TO/wav2vec2-base")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    csv_file = "wav2vec2_AASIST_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy", "Test Accuracy"])
    best_test_accuracy = 0.0
    best_model_path = "best_model_wav2vec2_AASIST.pth"
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
        correct = 0
        total = 0
        with torch.no_grad():
            for input_values, labels in tqdm(val_loader, desc="Validation"):
                input_values, labels = input_values.to(device), labels.to(device)
                outputs = model(input_values)
                val_loss = criterion(outputs.squeeze(), labels)
                total_val_loss += val_loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted.squeeze() == labels).sum().item()
                total += labels.size(0)
            val_accuracy = correct / total if total > 0 else 0

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        train_accuracy = evaluate(model, train_loader, device)
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, train_accuracy, val_accuracy, test_accuracy])

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Test Accuracy: {best_test_accuracy:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'wav2vec2_AASIST_{epoch+1}.pth')

if __name__ == "__main__":
    main()

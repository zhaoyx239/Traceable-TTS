import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from utils.preprocess import load_audio
from models.wav2vec_classifier import Wav2VecClassifier
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import wandb
from torch.optim.lr_scheduler import LinearLR

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
        input_values = self.processor(waveform.squeeze().numpy(), 
                                    sampling_rate=16000, 
                                    return_tensors="pt").input_values

        return input_values.squeeze(), torch.tensor(self.label, dtype=torch.float32)

def create_train_loader(real_data_dirs, synthetic_data_dirs, processor, batch_size=32):
    train_real_datasets = [AudioDataset(real_data_dir, 0, processor) for real_data_dir in real_data_dirs]
    train_synthetic_datasets = [AudioDataset(synthetic_data_dir, 1, processor) for synthetic_data_dir in synthetic_data_dirs]

    print("\nReal data directories and sample counts:")
    for i, dataset in enumerate(train_real_datasets):
        print(f"Directory {i+1}: {real_data_dirs[i]}")
        print(f"Number of samples: {len(dataset)}")
        wandb.log({f"real_data/dir_{i+1}_samples": len(dataset)})
    
    print("\nSynthetic data directories and sample counts:")
    for i, dataset in enumerate(train_synthetic_datasets):
        print(f"Directory {i+1}: {synthetic_data_dirs[i]}")
        print(f"Number of samples: {len(dataset)}")
        wandb.log({f"synthetic_data/dir_{i+1}_samples": len(dataset)})

    train_dataset = torch.utils.data.ConcatDataset(train_real_datasets + train_synthetic_datasets)
    total_samples = len(train_dataset)
    print(f"\nTotal training samples: {total_samples}")
    wandb.log({"total_training_samples": total_samples})
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def log_gradients(model, step):
    gradients = {
        'mean': [],
        'std': [],
        'max': [],
        'min': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            if 'classifier' in name:  
                wandb.log({
                    f"gradients/{name}_mean": grad.mean().item(),
                    f"gradients/{name}_std": grad.std().item(),
                    f"gradients/{name}_max": grad.abs().max().item(),
                }, step=step)
            
            gradients['mean'].append(grad.mean().item())
            gradients['std'].append(grad.std().item())
            gradients['max'].append(grad.abs().max().item())
            gradients['min'].append(grad.min().item())
    
    if gradients['mean']:  
        wandb.log({
            "gradients/mean": np.mean(gradients['mean']),
            "gradients/std": np.mean(gradients['std']),
            "gradients/max": np.max(gradients['max']),
            "gradients/min": np.min(gradients['min']),
        }, step=step)

def main(args):
    wandb.init(
        project="wav2vec-classifier",
        name=f"{args.version}_{args.name}",
        config={
            "wav2vec_version": args.version,
            "vocoder": args.vocoder,
            "batch_size": args.batchsize,
            "learning_rate": 5e-5, 
            "weight_decay": 1e-4,
            "warmup_steps": 3000,
            "epochs": 20,
            "grad_clip": 1.0,
        }
    )

    if args.version == "base":
        model_name = "facebook/wav2vec2-base"
        processor = Wav2Vec2Processor.from_pretrained(
            "/PATH/TO/wav2vec2-base"
        )
    elif args.version == "xlsr300m":
        model_name = "facebook/wav2vec2-xls-r-300m"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "/PATH/TO/wav2vec2-xls-r-300m"
        )
    elif args.version == "xlsr1b":
        model_name = "facebook/wav2vec2-xls-r-1b"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "/PATH/TO/wav2vec2-xls-r-1b"
        )

    print(f"Using wav2vec2 version: {args.version} ({model_name})")
    wandb.log({"model_name": model_name})
    
    real_train_data_dirs = [
        "/PATH/TO/DATA"
    ]
    synthetic_train_data_dirs = [
        "/PATH/TO/DATA"
    ]
    
    train_loader = create_train_loader(
        real_train_data_dirs, 
        synthetic_train_data_dirs, 
        processor, 
        batch_size=args.batchsize
    )

    model = Wav2VecClassifier(args.version)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        eps=1e-8
    )
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  
        end_factor=1.0,    
        total_iters=wandb.config.warmup_steps  
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    ckpt_dir = f"ckpts/{args.name}_{args.version}"
    model_path = os.path.join(ckpt_dir, "model_last.pth")
    best_model_path = os.path.join(ckpt_dir, "model_best.pth")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        print(f"Resuming training from step {global_step}")
    else:
        print("Training from scratch...")
        global_step = 0
        best_train_loss = float("inf")
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(wandb.config.epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs}")
        for input_values, labels in progress_bar:
            input_values, labels = input_values.to(device), labels.to(device)
            
            outputs = model(input_values)
            loss = criterion(outputs.squeeze(1), labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=wandb.config.grad_clip)

            if global_step % 100 == 0:
                log_gradients(model, global_step)
            
            optimizer.step()
            
            if global_step < wandb.config.warmup_steps:
                warmup_scheduler.step()
            
            global_step += 1
            total_train_loss += loss.item()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "step": global_step
            })
            
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
                "warmup_progress": min(global_step / wandb.config.warmup_steps, 1.0)
            }, step=global_step)
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_train_loss:.4f} | Best Loss: {best_train_loss:.4f}")
        
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "global_step": global_step,
            "best_train_loss": best_train_loss,
            "epoch": epoch + 1
        }, model_path)
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({
                "model_state_dict": model.module.state_dict(),
                "global_step": global_step,
                "best_train_loss": best_train_loss
            }, best_model_path)
            print(f"New best model saved at step {global_step}")
        
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "best_train_loss": best_train_loss
        }, step=global_step)
    
    print(f"\nTraining completed. Final model saved to:\n{model_path}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wav2vec2 classifier")
    parser.add_argument("--version", type=str, default="base", 
                       choices=["base", "xlsr300m","xlsr1b"],
                       help="Wav2Vec2 model version")
    parser.add_argument("--vocoder", type=str, default="vocos", 
                       choices=["vocos", "bigvgan", "f5"],
                       help="Vocoder type")
    parser.add_argument("--batchsize", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--name",type=str)
    args = parser.parse_args()
    

    print("\n" + "="*50)
    print(f"Starting training with config:")
    print(f"  - Model: wav2vec2 {args.version}")
    print(f"  - Vocoder: {args.vocoder}")
    print(f"  - Batch size: {args.batchsize}")
    print("="*50 + "\n")
    
    main(args)
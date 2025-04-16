import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from transformers import Wav2Vec2Processor
from models.wav2vec_classifier import Wav2VecClassifier
from utils.preprocess import load_audio
from tqdm import tqdm
from train import collate_fn, AudioDataset  
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# Calculate metrics (TP, TN, FP, FN)
def calculate_metrics(outputs, labels, threshold):
    predicted = (outputs > threshold).float()
    tp = ((predicted == 1) & (labels == 1)).sum().item()  # True Positive
    tn = ((predicted == 0) & (labels == 0)).sum().item()  # True Negative
    fp = ((predicted == 1) & (labels == 0)).sum().item()  # False Positive
    fn = ((predicted == 0) & (labels == 1)).sum().item()  # False Negative
    return tp, tn, fp, fn

# Evaluate model on test set
def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

    with torch.no_grad():
        for input_values, labels in tqdm(test_loader, desc="Testing"):
            input_values, labels = input_values.to(device), labels.to(device)
            outputs = model(input_values).squeeze()
            if outputs.dim()!=0:
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            tp, tn, fp, fn = calculate_metrics(outputs, labels, 0.5)  # Using 0.5 as threshold
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn

    # Calculate AUC
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    auc = roc_auc_score(all_labels, all_outputs)

    # Calculate EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    # Find optimal threshold
    accuracies = []
    for thresh in thresholds:
        accuracies.append(accuracy_score(all_labels, all_outputs > thresh))
    best_threshold = thresholds[np.argmax(accuracies)]

    # Recalculate metrics with optimal threshold
    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    tp_total, tn_total, fp_total, fn_total = calculate_metrics(all_outputs, all_labels, best_threshold)

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"True Positives: {tp_total}")
    print(f"True Negatives: {tn_total}")
    print(f"False Positives: {fp_total}")
    print(f"False Negatives: {fn_total}")
    acc = (tp_total+tn_total)/(tp_total+tn_total+fp_total+fn_total)
    print(f"ACC: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f}")
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    
    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("test.png")
    plt.close()

# Load model with DataParallel support
def load_model_with_module(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):  
            k = 'module.' + k  
        new_state_dict[k] = v
    model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    return model

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label, processor):
        self.files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.files.append(os.path.join(root, file))
        print(f"Found {len(self.files)} .wav files in {data_dir}")
        self.label = label
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform = load_audio(file_path)
        input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        return input_values.squeeze(), torch.tensor(self.label, dtype=torch.float32)

# Create data loaders with balanced sampling
def create_data_loaders(data_paths_label_1, data_paths_label_0, processor, batch_size=8):
    # Initialize datasets
    datasets_label_1 = [AudioDataset(path, 1, processor) for path in data_paths_label_1]
    datasets_label_0 = [AudioDataset(path, 0, processor) for path in data_paths_label_0]
    
    # Calculate number of label 1 samples
    num_label_1 = sum(len(dataset) for dataset in datasets_label_1)

    # Sample evenly from label 0 subsets
    num_label_0_subsets = len(datasets_label_0)
    samples_per_subset = num_label_1 // num_label_0_subsets

    # Create subsets for label 0
    subsets_label_0 = []
    for dataset in datasets_label_0:
        indices = np.random.choice(len(dataset), samples_per_subset, replace=False)
        subsets_label_0.append(Subset(dataset, indices))
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets_label_1 + subsets_label_0)

    print(f"Total samples: {len(combined_dataset)}")

    # Create data loader
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return data_loader

def main():
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained("/PATH/TO/wav2vec2-base")
    model = Wav2VecClassifier()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    model = load_model_with_module(model, '/PATH/TO/model_best.pth', device)
    model.to(device)

    # Test data paths
    test_dir = "test-clean" # dev-clean for train-acc | test-clean for test-acc
    edit = "vocos" # musan / volume / reverb / pitch / 0.8 / 1.2 / format-mp3 / format-wav / 8k / vocos / bigvgan
    data_paths_label_1 = [
        f"/PATH/TO/DATA"
    ]
    data_paths_label_0 = [
        f"/PATH/TO/DATA"
    ]

    # Create test data loader
    test_loader = create_data_loaders(data_paths_label_1, data_paths_label_0, processor, batch_size=16)

    # Run evaluation
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
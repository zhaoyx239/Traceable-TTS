import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7" 
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from models.mel_classifier import MelResNetClassifier
from train_mel import collate_fn, SimpleAudioDataset 
import torch
from utils.preprocess import load_audio
def calculate_metrics(outputs, labels, threshold):
    predicted = (outputs > threshold).float()
    tp = ((predicted == 1) & (labels == 1)).sum().item() 
    tn = ((predicted == 0) & (labels == 0)).sum().item()  
    fp = ((predicted == 1) & (labels == 0)).sum().item()  
    fn = ((predicted == 0) & (labels == 1)).sum().item() 
    return tp, tn, fp, fn

def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

    with torch.no_grad():
        for waves, labels in tqdm(test_loader, desc="Testing"):
            waves, labels = waves.to(device), labels.to(device)
            outputs = model(waves).squeeze()
            if outputs.dim() == 0: 
                outputs = outputs.unsqueeze(0)  
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

            tp, tn, fp, fn = calculate_metrics(outputs, labels, 0.5) 
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    auc = roc_auc_score(all_labels, all_outputs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    accuracies = []
    for thresh in thresholds:
        accuracies.append(accuracy_score(all_labels, all_outputs > thresh))
    best_threshold = thresholds[np.argmax(accuracies)]

    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    tp_total, tn_total, fp_total, fn_total = calculate_metrics(all_outputs, all_labels, best_threshold)

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"True Positives: {tp_total}")
    print(f"True Negatives: {tn_total}")
    print(f"False Positives: {fp_total}")
    print(f"False Negatives: {fn_total}")
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f}")
    
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
    def __init__(self, data_dir, label):
        self.files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.files.append(os.path.join(root, file))
        print(f"Found {len(self.files)} .wav files in {data_dir}")
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform = load_audio(file_path)
        return waveform.squeeze(), torch.tensor(self.label, dtype=torch.float32)

def create_test_data_loader(real_dirs, fake_dirs, batch_size=32):
    datasets_real = [AudioDataset(d, 0) for d in real_dirs]
    datasets_fake = [AudioDataset(d, 1) for d in fake_dirs]

    num_fake = sum(len(dataset) for dataset in datasets_fake)

    num_real_subsets = len(datasets_real)
    samples_per_subset = num_fake // num_real_subsets

    subsets_real = []
    for dataset in datasets_real:
        indices = np.random.choice(len(dataset), samples_per_subset, replace=False)
        subsets_real.append(Subset(dataset, indices))

    test_set = torch.utils.data.ConcatDataset(datasets_fake + subsets_real)

    print(f"Total samples: {len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    return test_loader

def main():
    model = MelResNetClassifier(input_type='audio') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_with_module(model, '/PATH/TO/model_best.pth', device)
    model.to(device)

    test_dir = "test-clean" 
    real_dirs = [
            f"/PATH/TO/DATA/{test_dir}"
    ]
    fake_dirs = [
        f"/PATH/TO/DATA/{test_dir}"
    ]

    test_loader = create_test_data_loader(real_dirs, fake_dirs, batch_size=32)

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
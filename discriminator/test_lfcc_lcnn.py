import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from models.lfcc_lcnn import LFCC_LCNN
from utils.preprocess import load_audio
from tqdm import tqdm
from train_lfcc_lcnn import create_data_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  

def calculate_metrics(outputs, labels):
    predicted = (outputs > 0.5).float()
    tp = ((predicted == 1) & (labels == 1)).sum().item()  # True Positive
    tn = ((predicted == 0) & (labels == 0)).sum().item()  # True Negative
    fp = ((predicted == 1) & (labels == 0)).sum().item()  # False Positive
    fn = ((predicted == 0) & (labels == 1)).sum().item()  # False Negative
    return tp, tn, fp, fn

def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

    with torch.no_grad():
        for input_values, labels in tqdm(test_loader, desc="Testing"):
            input_values, labels = input_values.to(device), labels.to(device)
            outputs = model(input_values).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

            tp, tn, fp, fn = calculate_metrics(outputs, labels)
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    auc = roc_auc_score(all_labels, all_outputs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    print(f"True Positives: {tp_total}")
    print(f"True Negatives: {tn_total}")
    print(f"False Positives: {fp_total}")
    print(f"False Negatives: {fn_total}")
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f}")

def load_model_with_module(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
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


def main():
    model = LFCC_LCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_with_module(model, '/PATH/TO/best_model_lfcc_lcnn.pth', device)
    model.to(device)

    real_train_data = "/PATH/TO/DATA"
    real_val_data = "/PATH/TO/DATA"
    real_test_data = "/PATH/TO/DATA"

    synthetic_train_data = "/PATH/TO/DATA"
    synthetic_val_data = "/PATH/TO/DATA"
    synthetic_test_data = "/PATH/TO/DATA"

    _, _, test_loader = create_data_loaders(
        real_train_data, real_val_data, real_test_data,
        synthetic_train_data, synthetic_val_data, synthetic_test_data,
        batch_size=32
    )

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()

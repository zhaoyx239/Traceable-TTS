import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 使用 GPU
import torch

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from transformers import Wav2Vec2Processor
from models.wav2vec_classifier import Wav2VecClassifier
from utils.preprocess import load_audio
from tqdm import tqdm
from train import collate_fn, AudioDataset  # 使用训练代码中的相关部分
#import matplotlib.pyplot as plt
#import pandas as pd

from sklearn.metrics import accuracy_score

# 计算指标（TP, TN, FP, FN）
def calculate_metrics(outputs, labels, threshold):
    predicted = (outputs > threshold).float()
    tp = ((predicted == 1) & (labels == 1)).sum().item()  # True Positive
    tn = ((predicted == 0) & (labels == 0)).sum().item()  # True Negative
    fp = ((predicted == 1) & (labels == 0)).sum().item()  # False Positive
    fn = ((predicted == 0) & (labels == 1)).sum().item()  # False Negative
    return tp, tn, fp, fn

# 计算测试集结果
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
            
            tp, tn, fp, fn = calculate_metrics(outputs, labels, 0.5)  # 这里暂时用 0.5
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn

    # 计算 AUC
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    auc = roc_auc_score(all_labels, all_outputs)

    # 计算 EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    # 选择最佳阈值
    accuracies = []
    for thresh in thresholds:
        accuracies.append(accuracy_score(all_labels, all_outputs > thresh))
    best_threshold = thresholds[np.argmax(accuracies)]

    # 使用最佳阈值重新计算指标
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
    
    '''# 绘制 ROC 曲线并保存为图片
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("test.png")  # 保存为图片
    plt.close()  # 关闭图像，避免在非交互环境中显示'''
# 加载模型
def load_model_with_module(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):  # 如果键名没有 'module.' 前缀
            k = 'module.' + k  # 添加 'module.' 前缀
        new_state_dict[k] = v
    model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    return model

# 更新后的数据集类，适应两级目录结构
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label, processor):
        # 修改为递归遍历两级文件夹
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

# 更新数据加载器函数
def create_data_loaders(data_paths_label_1, data_paths_label_0, processor, batch_size=8):
    # 初始化数据集
    datasets_label_1 = [AudioDataset(path, 1, processor) for path in data_paths_label_1]
    datasets_label_0 = [AudioDataset(path, 0, processor) for path in data_paths_label_0]
    
    # 计算 label 为 1 的样本数量
    num_label_1 = sum(len(dataset) for dataset in datasets_label_1)

    # 从 label 为 0 的每个子集中均匀提取样本
    num_label_0_subsets = len(datasets_label_0)
    samples_per_subset = num_label_1 // num_label_0_subsets

    # 从每个 label 为 0 的子集中提取样本
    subsets_label_0 = []
    for dataset in datasets_label_0:
        indices = np.random.choice(len(dataset), samples_per_subset, replace=False)
        subsets_label_0.append(Subset(dataset, indices))
    
    # 合并数据集
    combined_dataset = torch.utils.data.ConcatDataset(datasets_label_1 + subsets_label_0)

    print(f"Total samples: {len(combined_dataset)}")

    # 创建数据加载器
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return data_loader

def main():
    # 加载 processor 和模型
    processor = Wav2Vec2Processor.from_pretrained("/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/models/wav2vec2-base")
    model = Wav2VecClassifier()

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model_with_module(model, '/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/ckpts/baseline/model_best.pth', device)
    model.to(device)

    # 测试数据路径
    test_dir = "test-clean" # dev-clean for train-acc | test-clean for test-acc
    change = "vocos" # musan / volume / reverb / pitch / 0.8 / 1.2 / format-mp3 / format-wav / 8k / vocos / bigvgan
    data_paths_label_1 = [
        # f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/bigvgan/{test_dir}"
        # f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts-{change}/{test_dir}"
        f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/f5tts/origin_bigvgan/test-clean"
    ]
    data_paths_label_0 = [
        f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/{test_dir}",
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice/{test_dir}",
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice2/{test_dir}",
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/e2tts/{test_dir}"
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/cosyvoice/{change}/{test_dir}",
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/e2tts/{change}/{test_dir}",
        #f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/MM/cosyvoice2/{change}/{test_dir}",
    ]

    # 创建测试数据加载器
    test_loader = create_data_loaders(data_paths_label_1, data_paths_label_0, processor, batch_size=16)

    # 运行测试
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
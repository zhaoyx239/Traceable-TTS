import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 使用 GPU
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from models.mel_classifier import MelResNetClassifier
from train_mel import collate_fn, SimpleAudioDataset  # 使用训练代码中的相关部分
import torch
from utils.preprocess import load_audio
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
        for waves, labels in tqdm(test_loader, desc="Testing"):
            waves, labels = waves.to(device), labels.to(device)
            outputs = model(waves).squeeze()
            if outputs.dim() == 0:  # outputs 是标量
                outputs = outputs.unsqueeze(0)  # 转换为 1 维数组
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

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
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f}")
    
    '''# 绘制 ROC 曲线并保存为图片
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")  # 保存为图片
    plt.close()  # 关闭图像，避免在非交互环境中显示

    # 保存预测结果到CSV
    results_df = pd.DataFrame({
        'Sample': range(len(all_labels)),
        'Label': all_labels,
        'Prediction': all_outputs.numpy()
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("Prediction results saved to 'prediction_results.csv'")
'''
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
    def __init__(self, data_dir, label):
        # 修改为递归遍历两级文件夹
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
# 创建测试数据加载器
def create_test_data_loader(real_dirs, fake_dirs, batch_size=32):
    # 创建数据集
    datasets_real = [AudioDataset(d, 0) for d in real_dirs]
    datasets_fake = [AudioDataset(d, 1) for d in fake_dirs]

    # 计算正样本数量
    num_fake = sum(len(dataset) for dataset in datasets_fake)

    # 从每个负样本子集中均匀提取样本
    num_real_subsets = len(datasets_real)
    samples_per_subset = num_fake // num_real_subsets

    # 从每个负样本子集中提取样本
    subsets_real = []
    for dataset in datasets_real:
        indices = np.random.choice(len(dataset), samples_per_subset, replace=False)
        subsets_real.append(Subset(dataset, indices))

    # 合并数据集
    test_set = torch.utils.data.ConcatDataset(datasets_fake + subsets_real)

    print(f"Total samples: {len(test_set)}")

    # 创建数据加载器
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    return test_loader

def main():
    # 初始化模型
    model = MelResNetClassifier(input_type='audio')  # 输入类型为音频
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model_with_module(model, '/hpc_stor03/sjtu_home/yuxiang.zhao/classification_project/ckpts/baseline_mel/model_best.pth', device)
    model.to(device)

    # 测试数据路径
    test_dir = "test-clean" # dev-clean for train-acc | test-clean for test-acc
    real_dirs = [
        # f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/groundtruth/LibriTTS/{test_dir}",
        # f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice/{test_dir}",
        f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/cosyvoice2/{test_dir}",
        f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/e2tts/{test_dir}"
    ]
    fake_dirs = [
        f"/hpc_stor03/sjtu_home/yuxiang.zhao/data/infer/f5tts/{test_dir}"
    ]

    # 创建测试数据加载器
    test_loader = create_test_data_loader(real_dirs, fake_dirs, batch_size=32)

    # 运行测试
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
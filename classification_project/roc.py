import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 读取两个 CSV 文件
df_baseline = pd.read_csv("prediction_results_baseline.csv")
df_ours = pd.read_csv("prediction_results_ours.csv")

# 提取真实标签和预测值
y_true_baseline = df_baseline.iloc[:, 1]  # 第二列是真实标签
y_pred_baseline = df_baseline.iloc[:, 2]  # 第三列是预测值

y_true_ours = df_ours.iloc[:, 1]  # 第二列是真实标签
y_pred_ours = df_ours.iloc[:, 2]  # 第三列是预测值

# 计算 ROC 曲线和 AUC
fpr_baseline, tpr_baseline, _ = roc_curve(y_true_baseline, y_pred_baseline)
auc_baseline = roc_auc_score(y_true_baseline, y_pred_baseline)

fpr_ours, tpr_ours, _ = roc_curve(y_true_ours, y_pred_ours)
auc_ours = roc_auc_score(y_true_ours, y_pred_ours)

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, color="black", linestyle="--", label=f"Baseline (AUC = {0.8823:.4f})")
plt.plot(fpr_ours, tpr_ours, color="black", linestyle="-", label=f"Ours (AUC = {0.9421:.4f})")

# 添加图例
plt.legend(loc="lower right", fontsize=16)

# 添加标题和标签
plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)

# 增大坐标轴刻度字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 加粗坐标轴
for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(1)  # 设置坐标轴线宽为 1

# 保存图片
plt.savefig("roc_curves_comparison.png", bbox_inches="tight", dpi=300)
plt.close()
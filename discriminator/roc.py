import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

df_baseline = pd.read_csv("prediction_results_baseline.csv")
df_ours = pd.read_csv("prediction_results_ours.csv")

y_true_baseline = df_baseline.iloc[:, 1] 
y_pred_baseline = df_baseline.iloc[:, 2] 

y_true_ours = df_ours.iloc[:, 1] 
y_pred_ours = df_ours.iloc[:, 2]  

fpr_baseline, tpr_baseline, _ = roc_curve(y_true_baseline, y_pred_baseline)
auc_baseline = roc_auc_score(y_true_baseline, y_pred_baseline)

fpr_ours, tpr_ours, _ = roc_curve(y_true_ours, y_pred_ours)
auc_ours = roc_auc_score(y_true_ours, y_pred_ours)

plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, color="black", linestyle="--", label=f"Baseline (AUC = {0.8823:.4f})")
plt.plot(fpr_ours, tpr_ours, color="black", linestyle="-", label=f"Ours (AUC = {0.9421:.4f})")

plt.legend(loc="lower right", fontsize=16)

plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(1) 

plt.savefig("roc_curves_comparison.png", bbox_inches="tight", dpi=300)
plt.close()
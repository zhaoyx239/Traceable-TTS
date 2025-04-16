import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_distribution(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 分离正负样本
    positive_samples = df[df['Label'] == 1]['Prediction']
    negative_samples = df[df['Label'] == 0]['Prediction']

    # 设置绘图参数
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.001, 0.001)  # 设置横轴分度值为0.01

    # 绘制直方图
    plt.hist(positive_samples, bins=bins, alpha=0.5, label='Positive Class', density=True)
    plt.hist(negative_samples, bins=bins, alpha=0.5, label='Negative Class', density=True)

    # 添加图例和标签
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Positive and Negative Classes')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plt.savefig('prediction_distribution_ours.png')
    print("Distribution plot saved to 'prediction_distribution.png'")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    csv_file = 'prediction_results_ours.csv'
    plot_prediction_distribution(csv_file)
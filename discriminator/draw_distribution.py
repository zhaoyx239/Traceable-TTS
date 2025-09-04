import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_distribution(csv_file):
    df = pd.read_csv(csv_file)
    positive_samples = df[df['Label'] == 1]['Prediction']
    negative_samples = df[df['Label'] == 0]['Prediction']
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.001, 0.001) 
    plt.hist(positive_samples, bins=bins, alpha=0.5, label='Positive Class', density=True)
    plt.hist(negative_samples, bins=bins, alpha=0.5, label='Negative Class', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Positive and Negative Classes')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_distribution_ours.png')
    print("Distribution plot saved to 'prediction_distribution.png'")
    plt.show()
if __name__ == "__main__":
    csv_file = 'prediction_results_ours.csv'
    plot_prediction_distribution(csv_file)
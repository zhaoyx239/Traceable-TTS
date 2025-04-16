import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_distribution(csv_file):
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Separate positive and negative samples
    positive_samples = df[df['Label'] == 1]['Prediction']
    negative_samples = df[df['Label'] == 0]['Prediction']

    # Set plot parameters
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.001, 0.001)  # Set x-axis bins with 0.01 interval

    # Plot histograms
    plt.hist(positive_samples, bins=bins, alpha=0.5, label='Positive Class', density=True)
    plt.hist(negative_samples, bins=bins, alpha=0.5, label='Negative Class', density=True)

    # Add legend and labels
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Positive and Negative Classes')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig('prediction_distribution_ours.png')
    print("Distribution plot saved to 'prediction_distribution.png'")

    # Display plot
    plt.show()

if __name__ == "__main__":
    csv_file = 'prediction_results_ours.csv'
    plot_prediction_distribution(csv_file)
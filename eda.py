# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    """
    Perform exploratory data analysis on the given dataset.

    :param data: DataFrame containing the football match data
    """

    # Basic Descriptive Statistics
    print("Basic Descriptive Statistics:")
    print(data.describe())

    # Distribution of key features like goals, wins, losses
    print("\nDistribution of Key Features:")
    for column in ['goals', 'wins', 'losses']:  # Replace with actual column names
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    # Correlation Matrix
    print("\nCorrelation Matrix:")
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()


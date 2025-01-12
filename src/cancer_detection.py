"""
Breast Cancer Detection Model
----------------------------

This module implements a machine learning model for breast cancer detection using PCA.
It provides functionality for data preprocessing, feature extraction, and classification
of breast cancer cases as either Malignant (M) or Benign (B).

Author: [Your Name]
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def drop_nan_columns(df):
    """
    Drops columns containing NaN values in the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: A new DataFrame with columns containing NaN values removed.
    """
    print("Columns with missing values:")
    print(df.isna().sum())
    df_cleaned = df.dropna(axis=1)
    print(f"\nRemoved {len(df.columns) - len(df_cleaned.columns)} columns with missing values")
    return df_cleaned

def split_features_and_labels(df):
    """
    Splits the DataFrame into features and label vector for classification.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and a 'diagnosis' column.

    Returns:
        tuple: (features DataFrame, label array) where labels are encoded as:
            - 1 for Malignant (M)
            - 0 for Benign (B)
    """
    features = df.drop('diagnosis', axis=1)
    y = np.array(df.diagnosis.map({'M': 1, 'B': 0}))
    print(f"Dataset shape: {features.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return features, y

def perform_PCA(df, n_components=2):
    """
    Performs PCA on the input DataFrame to reduce dimensionality.

    Args:
        df (pd.DataFrame): The input feature DataFrame.
        n_components (int): Number of principal components to keep.

    Returns:
        tuple: (transformed data array, PCA model) containing:
            - Transformed feature matrix with reduced dimensions
            - Fitted PCA model for future transformations
    """
    # Scale the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Apply PCA
    pca_model = PCA(n_components=n_components)
    data_PCA = pca_model.fit_transform(df_scaled)
    
    print(f"Explained variance ratio: {pca_model.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca_model.explained_variance_ratio_):.2%}")
    
    return data_PCA, pca_model

def visualize_pca_results(X_pca, y, title="PCA Results"):
    """
    Creates a scatter plot of the PCA results, color-coded by diagnosis.

    Args:
        X_pca (np.ndarray): PCA-transformed feature matrix
        y (np.ndarray): Label vector (1 for Malignant, 0 for Benign)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                         cmap='RdYlBu', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """
    Main function to demonstrate the breast cancer detection workflow.
    """
    # Load the dataset
    print("Loading dataset...")
    cancer_data = pd.read_csv('Cancer_Data.csv')
    
    # Preprocess data
    print("\nCleaning data...")
    cancer_data_cleaned = drop_nan_columns(cancer_data)
    
    # Split features and labels
    print("\nPreparing features and labels...")
    X, y = split_features_and_labels(cancer_data_cleaned)
    
    # Perform PCA
    print("\nApplying PCA...")
    X_pca, pca_model = perform_PCA(X)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_pca_results(X_pca, y, "Breast Cancer Detection - PCA Results")

if __name__ == '__main__':
    main()

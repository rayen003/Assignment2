"""
Main Module
----------
Main script to run the breast cancer detection pipeline.
"""

import os
import pandas as pd
from preprocessing import drop_nan_columns, split_features_and_labels
from feature_extraction import perform_PCA
from visualization import visualize_pca_results

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'Cancer_Data.csv')

def main():
    """
    Main function to demonstrate the breast cancer detection workflow.
    """
    # Load the dataset
    print("Loading dataset...")
    cancer_data = pd.read_csv(DATA_FILE)
    
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

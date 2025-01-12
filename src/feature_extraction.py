"""
Feature Extraction Module
------------------------
Implements PCA and other feature extraction methods for dimensionality reduction.
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

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

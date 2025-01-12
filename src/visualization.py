"""
Visualization Module
------------------
Contains functions for visualizing the breast cancer detection results.
"""

import matplotlib.pyplot as plt
import seaborn as sns

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

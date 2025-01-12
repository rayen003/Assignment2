"""
Data Preprocessing Module
------------------------
Handles data cleaning and preparation for the breast cancer detection model.
"""

import pandas as pd
import numpy as np

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

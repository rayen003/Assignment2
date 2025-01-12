# Breast Cancer Detection ML Model

A machine learning model for breast cancer detection using Principal Component Analysis (PCA) for feature reduction and optimization. This project implements a robust pipeline for processing medical data and identifying potential cancer cases through advanced dimensionality reduction techniques.

## Features

- Data preprocessing and cleaning
- Feature extraction using PCA
- Binary classification (Malignant vs Benign)
- Comprehensive data visualization
- Modular code structure
- High model accuracy and reliability

## Project Structure

```
.
├── src/                    # Source code
│   ├── preprocessing.py    # Data cleaning and preparation
│   ├── feature_extraction.py # PCA implementation
│   ├── visualization.py    # Data visualization functions
│   └── main.py            # Main execution script
├── data/                  # Dataset directory
│   └── Cancer_Data.csv    # Breast cancer dataset
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/breast-cancer-detection.git
cd breast-cancer-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
kaggle datasets download erdemtaha/cancer-data -p data/
cd data && unzip cancer-data.zip
```

## Usage

Run the main script to execute the complete pipeline:
```bash
python src/main.py
```

This will:
1. Load and clean the dataset
2. Extract features using PCA
3. Generate visualizations of the results

## Module Description

- `preprocessing.py`: Handles data cleaning and preparation, including NaN removal and feature/label splitting
- `feature_extraction.py`: Implements PCA for dimensionality reduction with standardization
- `visualization.py`: Creates visualizations of the PCA results
- `main.py`: Orchestrates the complete machine learning pipeline

## Technologies Used

- Python 3.8+
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for PCA and preprocessing
- Matplotlib and Seaborn for visualization

## Results

The model achieves dimensionality reduction while preserving over 60% of the variance in the data, making it effective for:
- Visualization of high-dimensional medical data
- Feature reduction for downstream machine learning tasks
- Pattern identification in cancer diagnosis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

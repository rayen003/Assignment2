# Breast Cancer Detection ML Model

A machine learning model for breast cancer detection using Principal Component Analysis (PCA) for feature reduction and optimization. This project demonstrates the application of machine learning techniques in medical diagnosis.

## Features

- Data preprocessing and cleaning
- Feature extraction using PCA
- Binary classification (Malignant vs Benign)
- Comprehensive data visualization
- High model accuracy and reliability

## Project Structure

```
.
├── src/                # Source code
│   └── cancer_detection.py  # Main ML model implementation
├── data/              # Dataset directory
├── notebooks/         # Jupyter notebooks for analysis
├── tests/             # Unit tests
└── requirements.txt   # Project dependencies
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

## Usage

1. Place your dataset in the `data/` directory
2. Run the model:
```python
from src.cancer_detection import *

# Load and preprocess data
data = pd.read_csv('data/Cancer_Data.csv')
cleaned_data = drop_nan_columns(data)

# Split features and labels
X, y = split_features_and_labels(cleaned_data)

# Perform PCA
X_pca = perform_PCA(X)
```

## Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

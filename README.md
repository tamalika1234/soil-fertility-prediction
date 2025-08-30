# Soil Fertility Prediction Project

This project predicts soil fertility levels based on various soil parameters using a Random Forest machine learning model.

## Dataset

- The dataset `dataset1.csv` contains numeric features like N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B and a target column `Output` indicating soil fertility level.
- **Note:** CSV files and model files are ignored in GitHub via `.gitignore`.

## Features Used

- N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B

## Model

- Random Forest Classifier from scikit-learn
- Data preprocessing includes:
  - Filling missing numeric values with mean
  - Standard scaling of numeric features
  - Label encoding of the target column

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tamalika1234/soil-fertility-prediction.git

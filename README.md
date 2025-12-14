# Project Title

Multimodal Direct Photodisappearance Kinetics Models of Chemicals in Water 

## Project Overview

This project consists of six Python scripts designed to perform molecular data preprocessing, feature extraction, multimodal feature combination, model training and optimization, applicability domain characterization, and prediction of new data. The functionalities are as follows:

1. **Data Preprocessing**: Standardizes numerical features, one-hot encodes categorical features, and separates target values.
2. **Fingerprint Generation**: Generates multiple molecular fingerprint features (e.g., ECFP4, ECFP6, MACCS, AFP, PubChem).
3. **Feature Combination**: Integrates SMILES sequences, molecular graph structures, experimental conditions, and fingerprint features to create multimodal features.
4. **Model Training and Optimization**: Trains regression models using XGBoost, performs hyperparameter tuning, and evaluates model performance.
5. **Applicability Domain Characterization**: Characterizes the applicability domain based on feature-endpoint landscapes.
6. **New Data Prediction**: Predicts outcomes for new data using trained models and preprocessing pipelines.


## Script Descriptions

### 1_preprocessing.py

#### Functionality
Automates the data preprocessing pipeline, including numerical feature standardization, categorical feature one-hot encoding, and target value separation.

#### Input
- Raw data: raw_data.csv file (must include the following columns)

#### Output
- Preprocessed features: processed_features.csv
- Preprocessing pipeline: preprocessor_pipeline.joblib
- Target values file: target.csv

---

### 2_fingerprint.py

#### Functionality
Generates multiple molecular fingerprint features (ECFP4/ECFP6/MACCS/AFP/PubChem) in batch.

#### Input
- Preprocessed feature file: processed_features.csv
- Fingerprint configuration parameters:
  - ECFP4/ECFP6: Radius (2/3), fingerprint length (512)
  - MACCS: Fixed 167 dimensions
  - AFP: 512-dimensional Avalon fingerprint
  - PubChem: 881-dimensional Morgan fingerprint

#### Output
- Multiple fingerprint files (saved separately by type):
  - ECFP4_512.csv
  - ECFP6_512.csv
  - MACCS.csv
  - AFP_512.csv
  - PubChemFP.csv

---

### 3_feature_combined.py

#### Functionality
Integrates SMILES sequences, molecular graph structures, experimental conditions, and fingerprint features to create multimodal feature combinations.

#### Input
- Preprocessed features: processed_features.csv
- Fingerprint files: AFP_512.csv/ECFP4_512.csv/ECFP6_512.csv/MACCS.csv/PubChemFP.csv

#### Output
- Multimodal feature sets (*.npy):

---

### 4_XGBoost-all.py

#### Functionality
Trains and optimizes regression models using XGBoost, including hyperparameter tuning, cross-validation, model interpretation, and result saving.

#### Input
- Feature data: .npy files in the input_features directory (each file contains sample feature matrices)
- Target data: target.csv file (contains target values for all samples)
- Global parameter: RANDOM_STATE

#### Output
- Model files: results-xgb/*_best_model.joblib (best XGBoost model)
- Prediction data:
  - *_best_train_data.csv
  - *_best_test_data.csv
- Evaluation metrics:
  - *_runs.csv (detailed metrics for 10 runs)
  - XGBoost_average_results.csv (average metrics across files)
  - *_shap_values.csv

---

### 5_ADFEL.py

#### Functionality
ADFRL: Application Domain Characterization Method Based on Feature-Endpoint Topography
## Input
    - Training data: train_data.csv
    - Test data: test_data.csv
    - Global parameters:
      - a=25
      - W_BINARY_PART = 0.5

## Output
    - Application domain metrics:
      - AD_Metrics_a=25.csv
      - Validation results in the result_AD/ directory

---

### 6_make_predictions.py

#### Functionality
Predicts outcomes for new data using trained models and preprocessing pipelines.

#### Input
- New data file: NEW.csv (contains SMILES codes and feature data)
- MACCS fingerprint file: MACCS_NEW.csv (molecular fingerprint data)
- Pretrained model files:
  - preprocessor_pipeline.joblib (preprocessing pipeline)
  - model.joblib (trained XGBoost model)

#### Output
- predictions.csv (contains two columns: original SMILES codes and prediction results)
---

## Environment Requirements

This project requires the following Python packages. Ensure that the specified versions are installed in your environment:

- **pandas**: 2.0.3
- **scikit-learn**: 1.3.2
- **joblib**: 1.4.2
- **numpy**: 1.24.3
- **rdkit**: 2022.9.5
- **deepchem**: 2.8.0
- **xgboost**: 2.1.1
- **optuna**: 3.6.1
- **matplotlib**: 3.7.5
- **shap**: 0.44.1
- **scipy**: 1.10.1

### Installation Guide
To set up the environment, follow these steps:

1. **Create a Virtual Environment (Recommended)**

   - conda create -n kd_env python=3.8
   - conda activate kd_env

2. **Install Required Packages**

  Install the packages using pip since they are sourced from PyPI:
  - pip install pandas==2.0.3
  - pip install scikit-learn==1.3.2
  - pip install joblib==1.4.2
  - pip install numpy==1.24.3
  - pip install rdkit-pypi==2022.9.5
  - pip install deepchem==2.8.0
  - pip install xgboost==2.1.1
  - pip install optuna==3.6.1
  - pip install matplotlib==3.7.5
  - pip install shap==0.44.1
  - pip install scipy==1.10.1




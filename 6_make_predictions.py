# 6_make_predictions.py
"""
## Function Overview
Use the trained model and preprocessing pipeline to make predictions on new data.

## Input
  - New data file: NEW.csv or NEW.xlsx (contains SMILES codes and feature data)
  - Pre-saved model files: 
    * preprocessor_pipeline.joblib (preprocessing pipeline)
    * model.joblib (trained XGBoost model)

## Output
  - predictions.csv (contains two columns: original SMILES codes and prediction results)
  
@author: HeJL_DLUT
"""

import pandas as pd
import numpy as np
import joblib
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit import RDLogger

# Function to generate MACCS fingerprints for a single SMILES string
def generate_maccs_fingerprints(smiles):
    """
    Generates a 167-bit MACCS fingerprint vector for a given SMILES string.
    Returns a zero vector if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((167,), dtype=int)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)
    return maccs_array

# --- Main Script ---

# Disable RDKit warning logs for cleaner output
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Load the saved preprocessing pipeline and model
preprocessor = joblib.load('preprocessor_pipeline.joblib') 
model = joblib.load('model.joblib')

# Auto-detect and load new data from .xlsx or .csv
new_data_path_xlsx = 'NEW.xlsx'
new_data_path_csv = 'NEW.csv'

if os.path.exists(new_data_path_xlsx):
    print(f"Found '{new_data_path_xlsx}', converting to CSV for processing.")
    # Read from Excel and save to CSV, ensuring UTF-8 encoding
    excel_df = pd.read_excel(new_data_path_xlsx)
    excel_df.to_csv(new_data_path_csv, index=False, encoding='utf-8-sig')
    new_data_path = new_data_path_csv
elif os.path.exists(new_data_path_csv):
    print(f"Found '{new_data_path_csv}', loading data.")
    new_data_path = new_data_path_csv
else:
    raise FileNotFoundError("No 'NEW.csv' or 'NEW.xlsx' file found in the directory.")

# Load new data from the determined CSV path
try:
    new_df = pd.read_csv(new_data_path, encoding='utf-8')
except UnicodeDecodeError:
    new_df = pd.read_csv(new_data_path, encoding='gbk')

# Extract numerical features and categorical features (needed for column name retrieval)
num_features = new_df.iloc[:, [1, 3, 7, 8, 9]].copy()
cat_features = new_df.iloc[:, [5, 6]].copy()

# Transform new data using the preprocessing pipeline
processed_features = preprocessor.transform(new_df)

# Generate MACCS fingerprints from SMILES in the new data on-the-fly
print("Generating MACCS fingerprints for new data...")
smiles_list = new_df['SMILES'].tolist()
maccs_new_features = np.array([generate_maccs_fingerprints(s) for s in smiles_list])
print("Fingerprint generation complete.")

# Combine condition features and MACCS fingerprints
condition_features = processed_features
if condition_features.shape[0] != maccs_new_features.shape[0]:
    raise ValueError("The number of samples in condition features and MACCS fingerprints is inconsistent.")
X_combined_new = np.hstack((condition_features, maccs_new_features))

# Check if the number of features matches those used during training
expected_num_features = model.get_booster().num_features()
if X_combined_new.shape[1] != expected_num_features:
    raise ValueError(f"Feature count mismatch: expected {expected_num_features}, but got {X_combined_new.shape[1]}")

# Use the trained model to make predictions
y_pred = model.predict(X_combined_new)

# Save prediction results along with SMILES codes
predictions_df = pd.DataFrame({'SMILES': new_df['SMILES'], 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

print("Prediction results have been saved to 'predictions.csv'.")
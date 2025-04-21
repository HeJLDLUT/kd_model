# 6_make_predictions.py
"""
## Function Overview
Use the trained model and preprocessing pipeline to make predictions on new data.

## Input
  - New data file: NEW.csv (contains SMILES codes and feature data)
  - MACCS fingerprint file: MACCS_NEW.csv (molecular fingerprint data)
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


# Load the saved preprocessing pipeline and model
preprocessor = joblib.load('preprocessor_pipeline.joblib') 
model = joblib.load('model.joblib')

# Load new data and specify encoding
new_data_path = 'NEW.csv'
try:
    new_df = pd.read_csv(new_data_path, encoding='utf-8')
except UnicodeDecodeError:
    # If utf-8 fails, try another encoding
    new_df = pd.read_csv(new_data_path, encoding='gbk')  # Adjust based on actual situation


# Extract numerical features and categorical features
num_features = new_df.iloc[:, [1, 3, 7, 8, 9]].copy()  # Ensure column indices match those used during training
cat_features = new_df.iloc[:, [5, 6]].copy()  # Ensure column indices match those used during training

# Transform new data using the preprocessing pipeline (excluding SMILES codes)
processed_features = preprocessor.transform(new_df)

# Get column names for one-hot encoded categorical features
ohe_categories = preprocessor.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features.columns)
all_feature_names = num_features.columns[1:].tolist() + ohe_categories.tolist()  # Exclude SMILES codes

# Convert to DataFrame
processed_df = pd.DataFrame(processed_features, columns=all_feature_names)
processed_df.insert(0, 'SMILES', new_df['SMILES']) 


# Generate new MACCS fingerprints
maccs_new_path = 'MACCS_NEW.csv'
maccs_new_df = pd.read_csv(maccs_new_path)

# Process MACCS fingerprint data
maccs_new_df = maccs_new_df.apply(pd.to_numeric, errors='coerce').fillna(0)
maccs_new_features = maccs_new_df.values 

condition_features = processed_features  

# Ensure the number of samples in condition features and MACCS features is consistent
if condition_features.shape[0] != maccs_new_features.shape[0]:
    raise ValueError("The number of samples in condition features and MACCS fingerprints is inconsistent.")

# Combine condition features and MACCS fingerprints
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

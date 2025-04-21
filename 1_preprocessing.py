# 1_preprocessing.py
"""
## Function Overview
Implement a data preprocessing pipeline that automatically performs numerical feature standardization, categorical feature one-hot encoding, and separates and saves the target value column.

## Input
    - Raw data: `raw_data.csv` file (must include the following columns):
      - Column 1: SMILES code
      - Columns 2/3/7/8/9: Numerical features
      - Columns 5/6: Categorical features
      - Column 11: Target value

## Output
    - Preprocessed features: `processed_features.csv`
    - Preprocessing pipeline: `preprocessor_pipeline.joblib`
    - Target value file: `target.csv`

@author: HeJL_DLUT
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  

# Input and output file paths
input_file_path = "raw_data.csv"
output_file_path = "processed_features.csv"

# Read the data
df = pd.read_csv(input_file_path)

# Extract SMILES code and required numerical features
num_features = df.iloc[:, [1, 3, 7, 8, 9]].copy()  # Column 1 is SMILES, the rest are numerical features

# Extract categorical features
cat_features = df.iloc[:, [5, 6]].copy()

# Create transformers for numerical and categorical features
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features.columns[1:].tolist()),  # Numerical feature columns excluding SMILES code
        ('cat', cat_transformer, cat_features.columns.tolist())  # Categorical feature columns
    ])

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply the pipeline to the feature data
processed_data = pipeline.fit_transform(df)

# Get column names for categorical features from the one-hot encoder
ohe_categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features.columns)

# Combine numerical feature and one-hot encoded categorical feature column names
all_feature_names = num_features.columns[1:].tolist() + ohe_categories.tolist()  # Excluding SMILES code

# Convert the processed data to a DataFrame
processed_df = pd.DataFrame(processed_data, columns=all_feature_names)  # Directly use processed_data

# Add the SMILES code to the processed DataFrame
processed_df.insert(0, 'SMILES', df.iloc[:, 1])  # Use column 2 as SMILES code

# Save to a new file
processed_df.to_csv(output_file_path, index=False)

print("The processed features have been saved to:", output_file_path)

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessor_pipeline.joblib')
print("Preprocessing pipeline has been saved to: preprocessor_pipeline.joblib")

# Input and output file paths
input_file_path = "raw_data.csv"
output_file_path = "target.csv"

# Read the data
df = pd.read_csv(input_file_path)

# Extract the target value column
target = df.iloc[:, 11]

# Save this column to a new file
target.to_csv(output_file_path, index=False, header=['Target'])  

print(f"The target column has been saved to: {output_file_path}")

# 4_XGBoost-all_random_split.py
"""
## Function Overview
Use the XGBoost regression model for automated training and optimization. 
This version uses a standard random 80:20 train/test split strategy.
The model's stability is evaluated over 10 different random splits.

## Input
    - Raw data: raw_data.csv (for splitting logic and ID alignment)
    - Feature data: .npy files in the input_features directory
    - Target data: target.csv file
    - Global parameter: RANDOM_STATE

## Output
    - Model file: results-xgb/*_best_model.joblib (best XGBoost model from 10 runs)
    - Prediction data:
      - *_best_train_data.csv
      - *_best_validation_data.csv
    - Evaluation metrics:
      - *_runs.csv (detailed metrics for 10 runs)
      - XGBoost_average_results.csv (average metrics for each file)
    - *_shap_values.csv (from the best run)

@author: HeJL_DLUT
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import optuna
import matplotlib.pyplot as plt
import os
import glob
import joblib
import shap
from rdkit import Chem

# Set global parameters
RANDOM_STATE = **
np.random.seed(RANDOM_STATE)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Handle result directory
result_folder = "results-xgb"
os.makedirs(result_folder, exist_ok=True)

# Result data structure
results = pd.DataFrame(columns=[
    'file', 'cv_mae', 'cv_rmse', 'cv_r2',
    'mae_train', 'rmse_train', 'r2_train',
    'mae_validation', 'rmse_validation', 'r2_validation'
])

def get_valid_indices_and_data():
    """
    Reads raw data, validates SMILES, and returns the indices of valid rows
    and the corresponding filtered raw data. This ensures alignment.
    """
    print("Aligning data by validating SMILES from raw_data.csv...")
    df_raw = pd.read_csv('raw_data.csv')
    smiles_list = df_raw.iloc[:, 1].tolist() 
    
    valid_indices = [i for i, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(str(smiles)) is not None]
    
    df_raw_valid = df_raw.iloc[valid_indices].reset_index(drop=True)
    print(f"Found {len(valid_indices)} valid samples, which will be used for splitting.")
    return valid_indices, df_raw_valid

def cross_validate(model, X, y):
    """Perform 10-fold cross-validation and return three metrics."""
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        
        mae_scores.append(mean_absolute_error(y_val_cv, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
        r2_scores.append(r2_score(y_val_cv, y_pred))
    
    return (np.mean(mae_scores), np.mean(rmse_scores), np.mean(r2_scores))

def objective(trial, X_train, y_train):
    """Optuna optimization objective function."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 0.5, log=True),
        'random_state': RANDOM_STATE
    }
    
    model = XGBRegressor(**params)
    cv_mae, cv_rmse, cv_r2 = cross_validate(model, X_train, y_train)
    
    trial.set_user_attr('cv_mae', cv_mae)
    trial.set_user_attr('cv_rmse', cv_rmse)
    return cv_r2  

# Perform data alignment ONCE before the main loop
valid_indices, df_raw_valid = get_valid_indices_and_data()

# Main loop over feature files
for file_path in glob.glob(os.path.join("input_features", "*.npy")):
    file_results = []
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    
    try:
        X_full = np.load(file_path)
        y_full = pd.read_csv('target.csv').values.flatten()
        
        # Filter data based on valid SMILES indices
        X = X_full[valid_indices]
        y = y_full[valid_indices]
        sample_ids = df_raw_valid.index.values + 1
    except Exception as e:
        print(f"Data loading or alignment failed: {e}")
        continue

    # Initialize best run record for this file
    best_run_info = {
        'run': None, 'r2_validation': -np.inf, 'model': None,
        'y_train_pred': None, 'y_val_pred': None,
        'X_train': None, 'X_val': None, 'y_train': None, 'y_val': None,
        'id_train': None, 'id_val': None
    }

    # Loop for 10 runs with different random splits
    for run in range(1, 11):
        print(f"\nRun {run}/10")
        
        # --- Standard Random Split 80:20 ---
        # Splitting X, y, and sample_ids simultaneously
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X, y, sample_ids,
            test_size=0.2,
            random_state=RANDOM_STATE + run  # Change seed for each run
        )
        
        # Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
        
        best_params = study.best_trial.params
        best_model = XGBRegressor(**best_params, random_state=RANDOM_STATE)
        best_model.fit(X_train, y_train)
        
        # Calculate metrics for this run
        metrics = {}
        metrics.update({
            'cv_mae': study.best_trial.user_attrs['cv_mae'],
            'cv_rmse': study.best_trial.user_attrs['cv_rmse'],
            'cv_r2': study.best_trial.value
        })
        
        y_train_pred = best_model.predict(X_train)
        metrics.update({
            'mae_train': mean_absolute_error(y_train, y_train_pred),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2_train': r2_score(y_train, y_train_pred)
        })
        
        y_val_pred = best_model.predict(X_val)
        metrics.update({
            'mae_validation': mean_absolute_error(y_val, y_val_pred),
            'rmse_validation': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'r2_validation': r2_score(y_val, y_val_pred)
        })
        
        # Check if this is the current best run
        if metrics['r2_validation'] > best_run_info['r2_validation']:
            best_run_info.update({
                'run': run, 'r2_validation': metrics['r2_validation'], 'model': best_model,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred,
                'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val,
                'id_train': id_train, 'id_val': id_val
            })
        
        # Save results for this run
        file_results.append(pd.DataFrame({'file': [os.path.basename(file_path)], **metrics}))
    
    # After 10 runs, save the best run's model and data
    if best_run_info['model'] is not None:
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save model
        model_filename = os.path.join(result_folder, f"{file_base_name}_best_model.joblib")
        joblib.dump(best_run_info['model'], model_filename)
        print(f"\nBest model from run {best_run_info['run']} (Validation R2={best_run_info['r2_validation']:.4f}) saved to: {model_filename}")
        
        # Create and save best training data CSV
        train_df = pd.DataFrame(best_run_info['X_train'], columns=[f'Feature_{i}' for i in range(1, best_run_info['X_train'].shape[1] + 1)])
        train_df.insert(0, 'SampleID', best_run_info['id_train'])
        train_df['Target'] = best_run_info['y_train']
        train_df['Predicted'] = best_run_info['y_train_pred']
        train_df.to_csv(os.path.join(result_folder, f"{file_base_name}_best_train_data.csv"), index=False)
        
        # Create and save best validation data CSV
        val_df = pd.DataFrame(best_run_info['X_val'], columns=[f'Feature_{i}' for i in range(1, best_run_info['X_val'].shape[1] + 1)])
        val_df.insert(0, 'SampleID', best_run_info['id_val'])
        val_df['Target'] = best_run_info['y_val']
        val_df['Predicted'] = best_run_info['y_val_pred']
        val_df.to_csv(os.path.join(result_folder, f"{file_base_name}_best_validation_data.csv"), index=False)
        print(f"Best training/validation data saved for {os.path.basename(file_path)}")

        # SHAP values for the best run
        X_combined = np.vstack((best_run_info['X_train'], best_run_info['X_val']))
        y_combined = np.hstack((best_run_info['y_train'], best_run_info['y_val']))
        id_combined = np.hstack((best_run_info['id_train'], best_run_info['id_val']))
        
        explainer = shap.Explainer(best_run_info['model'])
        shap_values = explainer(X_combined)
        
        shap_df = pd.DataFrame(shap_values.values, columns=[f'SHAP_Feature_{i}' for i in range(1, X_combined.shape[1] + 1)])
        shap_df.insert(0, 'SampleID', id_combined)
        shap_df['Target'] = y_combined
        
        shap_filename = os.path.join(result_folder, f"{file_base_name}_shap_values.csv")
        shap_df.to_csv(shap_filename, index=False)
        print(f"SHAP values from the best run have been saved to: {shap_filename}")
    
    # Combine results of the 10 runs and save
    if file_results:
        df_file = pd.concat(file_results, ignore_index=True)
        df_file.to_csv(os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_runs.csv"), index=False)
        
        # Calculate averages and append to the main results DataFrame
        numeric_cols = df_file.select_dtypes(include=np.number).columns
        avg_results = df_file[numeric_cols].mean().to_frame().T
        avg_results['file'] = os.path.basename(file_path)
        results = pd.concat([results, avg_results], ignore_index=True)

# Save final average results across all files
results.to_csv(os.path.join(result_folder, 'XGBoost_average_results.csv'), index=False)
print("\nAll processing completed!")
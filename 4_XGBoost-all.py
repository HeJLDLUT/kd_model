# 4_XGBoost-all.py
"""
## Function Overview
Use the XGBoost regression model for automated training and optimization, including hyperparameter tuning, cross-validation, model interpretation, and result saving.

## Input
    - Feature data: .npy files in the input_features directory (each file contains a sample feature matrix)
    - Target data: target.csv file (contains target values for all samples)
    - Global parameter: RANDOM_STATE

## Output
    - Model file: results-xgb/*_best_model.joblib (best XGBoost model)
    - Prediction data:
      - *_best_train_data.csv
      - *_best_test_data.csv
    - Evaluation metrics:
      - *_runs.csv (detailed metrics for 10 runs)
      - XGBoost_average_results.csv (average metrics for each file)
    - *_shap_values.csv

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

# Set global parameters
RANDOM_STATE = *
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
    'mae_test', 'rmse_test', 'r2_test'
])

def cross_validate(model, X, y):
    """Perform 10-fold cross-validation and return three metrics."""
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val = X[train_idx], X[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val)
        
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_scores.append(r2_score(y_val, y_pred))
    
    return (
        np.mean(mae_scores),
        np.mean(rmse_scores),
        np.mean(r2_scores)
    )

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
    
    # Save cross-validation metrics
    trial.set_user_attr('cv_mae', cv_mae)
    trial.set_user_attr('cv_rmse', cv_rmse)
    return cv_r2  

# Main loop
for file_path in glob.glob(os.path.join("input_features", "*.npy")):
    file_results = []
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    
    try:
        X = np.load(file_path)
        y = pd.read_csv('target.csv').values.flatten()
        sample_ids = np.arange(1, len(y) + 1)  
    except Exception as e:
        print(f"Data loading failed: {e}")
        continue

    # Initialize best run record
    best_run_info = {
        'run': None,
        'r2_test': -np.inf, 
        'model': None,
        'y_train_pred': None,
        'y_test_pred': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'id_train': None, 
        'id_test': None   
    }

    for run in range(1, 11):
        print(f"\nRun {run}/10")
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, sample_ids, test_size=0.2, random_state=RANDOM_STATE+run)
        
        # Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        
        # Get best parameters
        best_params = study.best_trial.params
        best_model = XGBRegressor(**best_params, random_state=RANDOM_STATE)
        best_model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = {}
        
        # Cross-validation metrics
        metrics.update({
            'cv_mae': study.best_trial.user_attrs['cv_mae'],
            'cv_rmse': study.best_trial.user_attrs['cv_rmse'],
            'cv_r2': study.best_trial.value
        })
        
        # Training set metrics
        y_train_pred = best_model.predict(X_train)
        metrics.update({
            'mae_train': mean_absolute_error(y_train, y_train_pred),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2_train': r2_score(y_train, y_train_pred)
        })
        
        # Test set metrics
        y_test_pred = best_model.predict(X_test)
        metrics.update({
            'mae_test': mean_absolute_error(y_test, y_test_pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2_test': r2_score(y_test, y_test_pred)
        })
        
        # Check if this is the current best run
        if metrics['r2_test'] > best_run_info['r2_test']:
            best_run_info.update({
                'run': run,
                'r2_test': metrics['r2_test'],
                'model': best_model,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'id_train': id_train, 
                'id_test': id_test    
            })
        
        # Save results
        file_results.append(pd.DataFrame({
            'file': [os.path.basename(file_path)],
            **metrics
        }))
    
    # Save the best run's model and data
    if best_run_info['model'] is not None:
        # Save model
        model_filename = os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_best_model.joblib")
        joblib.dump(best_run_info['model'], model_filename)
        print(f"\nBest model (R2={best_run_info['r2_test']:.4f}) saved to: {model_filename}")
        
        # Create training set DataFrame and add SampleID
        train_data = np.hstack((
            best_run_info['X_train'], 
            best_run_info['y_train'].reshape(-1, 1), 
            best_run_info['y_train_pred'].reshape(-1, 1)
        ))
        train_df = pd.DataFrame(train_data, 
                               columns=[f'Feature_{i}' for i in range(1, best_run_info['X_train'].shape[1] + 1)] + 
                               ['Target', 'Predicted'])
        train_df.insert(0, 'SampleID', best_run_info['id_train'])
        
        # Create test set DataFrame and add SampleID
        test_data = np.hstack((
            best_run_info['X_test'], 
            best_run_info['y_test'].reshape(-1, 1), 
            best_run_info['y_test_pred'].reshape(-1, 1)
        ))
        test_df = pd.DataFrame(test_data, 
                               columns=[f'Feature_{i}' for i in range(1, best_run_info['X_test'].shape[1] + 1)] + 
                               ['Target', 'Predicted'])
        test_df.insert(0, 'SampleID', best_run_info['id_test']) 
        
        # Save DataFrame as CSV files
        train_df.to_csv(os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_best_train_data.csv"), index=False)
        test_df.to_csv(os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_best_test_data.csv"), index=False)
        print(f"Best training/validation data saved for {os.path.basename(file_path)}")
        

        X_combined = np.vstack((best_run_info['X_train'], best_run_info['X_test']))
        y_combined = np.hstack((best_run_info['y_train'], best_run_info['y_test']))
        id_combined = np.hstack((best_run_info['id_train'], best_run_info['id_test'])) 
        
        # Create SHAP explainer
        explainer = shap.Explainer(best_run_info['model'])
        shap_values = explainer(X_combined)  # Calculate SHAP values for all samples
        
        # Convert SHAP values to DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=[f'SHAP_Feature_{i}' for i in range(1, X_combined.shape[1] + 1)])
        
        # Add SampleID and target values
        shap_df.insert(0, 'SampleID', id_combined) 
        shap_df['Target'] = y_combined
        
        # Save SHAP values to CSV file
        shap_filename = os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_shap_values.csv")
        shap_df.to_csv(shap_filename, index=False)
        print(f"SHAP values have been saved to: {shap_filename}")
    
    # Combine results and save
    if file_results:
        df_file = pd.concat(file_results, ignore_index=True)
        df_file.to_csv(
            os.path.join(result_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_runs.csv"),
            index=False
        )
        
        # Calculate averages (numeric columns only)
        numeric_cols = df_file.select_dtypes(include=np.number).columns
        avg_results = df_file[numeric_cols].mean().to_frame().T
        avg_results['file'] = os.path.basename(file_path)
        results = pd.concat([results, avg_results], ignore_index=True)

# Save final results
results.to_csv(os.path.join(result_folder, 'XGBoost_average_results.csv'), index=False)
print("\nAll processing completed!")

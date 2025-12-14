# 5-2_ADFRL.py
"""
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

@author: HeJL_DLUT
"""

# ==================== Import Libraries ====================
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cdist
import warnings
import os
import glob

# Ensure the result folder exists
os.makedirs('result_AD', exist_ok=True)

warnings.filterwarnings('ignore')

# ==================== Global Parameter Configuration ====================
# Exponential weight parameter (controls similarity decay rate)
a = 25

# Weight for combining the two Tanimoto scores
# W_BINARY_PART determines the contribution of the binary features (fingerprints).
# The contribution of the continuous features will be (1 - W_BINARY_PART).
W_BINARY_PART = 0.5

# ==================== Data Preprocessing Module ====================
def load_and_preprocess_data():
    # Load raw data (Modified to auto-detect from results-xgb)
    result_dir = "results-xgb"
    train_files = glob.glob(os.path.join(result_dir, "*_best_train_data.csv"))
    if not train_files:
        raise FileNotFoundError(f"No training data was found.")
    latest_train_file = max(train_files, key=os.path.getmtime)
    latest_test_file = latest_train_file.replace("_best_train_data.csv", "_best_validation_data.csv")
    
    if not os.path.exists(latest_test_file):
        raise FileNotFoundError(f"The corresponding validation set file was not found: {latest_test_file}")

    print(f"Training data is being loaded: {latest_train_file}")
    print(f"Training data is being loaded: {latest_test_file}")

    df_train = pd.read_csv(latest_train_file, index_col='SampleID')
    df_test = pd.read_csv(latest_test_file, index_col='SampleID')
    
    # Define feature columns based on the new split requirement
    # Continuous: 1 to 12
    continuous_features = [f'Feature_{i}' for i in range(1, 13)]
    # Binary: 13 to 179
    binary_features = [f'Feature_{i}' for i in range(13, 180)]
    
    all_features = continuous_features + binary_features
    
    # Ensure all features are numeric
    df_train[all_features] = df_train[all_features].apply(pd.to_numeric, errors='coerce')
    df_test[all_features] = df_test[all_features].apply(pd.to_numeric, errors='coerce')
    
    # Check for NaN values and fill with 0
    nan_train = df_train[all_features].isnull().sum().sum()
    nan_test = df_test[all_features].isnull().sum().sum()
    if nan_train > 0 or nan_test > 0:
        print(f"Warning: NaN values detected. Training NaNs: {nan_train}, Test NaNs: {nan_test}. Filling with 0.")
        df_train[all_features] = df_train[all_features].fillna(0)
        df_test[all_features] = df_test[all_features].fillna(0)
    
    # Split data into numpy arrays
    X_train_cont = df_train[continuous_features].values
    X_train_bin = df_train[binary_features].values
    X_test_cont = df_test[continuous_features].values
    X_test_bin = df_test[binary_features].values

    print("Data loaded and split:")
    print(f" - Continuous features (Vector Tanimoto): {len(continuous_features)} (Feature_1 to Feature_12)")
    print(f" - Binary features (Set Tanimoto): {len(binary_features)} (Feature_13 to Feature_179)")
    
    return df_train, df_test, X_train_cont, X_train_bin, X_test_cont, X_test_bin

# ==================== Core Similarity Calculation Class ====================
class HybridTanimotoCalculator:
    """
    Hybrid Tanimoto Similarity Calculator
    
    Scientific Rationale:
    1. Continuous Features: Uses 'Vector Tanimoto' (also known as Tanimoto Coefficient for continuous variables).
       Formula: (A . B) / (|A|^2 + |B|^2 - A . B)
    2. Binary Features: Uses standard Tanimoto (Jaccard Index).
       Formula: |A n B| / |A u B|
    """
    
    def __init__(self, X_train_cont, X_train_bin, weight_binary=0.5):
        self.X_train_cont = X_train_cont
        self.X_train_bin = X_train_bin.astype(bool) # Optimize for Jaccard
        self.w_binary = weight_binary
        self.w_cont = 1 - weight_binary
        
        self.train_cont_norm_sq = np.sum(self.X_train_cont ** 2, axis=1)
        
        print(f"HybridTanimotoCalculator initialized. Binary Weight: {self.w_binary:.2f}, Continuous Weight: {self.w_cont:.2f}")

    def _calculate_vector_tanimoto(self, X_query):
        """
        Calculates Vector Tanimoto Similarity for continuous variables.
        S(A, B) = (A . B) / (|A|^2 + |B|^2 - A . B)
        """
        # Dot product: (n_query, n_train)
        dot_product = np.dot(X_query, self.X_train_cont.T)
        
        # Query norms squared: (n_query,)
        query_norm_sq = np.sum(X_query ** 2, axis=1)
        
        # Denominator: |A|^2 + |B|^2 - A.B
        # Broadcasting: (n_query, 1) + (1, n_train) - (n_query, n_train)
        denominator = query_norm_sq[:, np.newaxis] + self.train_cont_norm_sq[np.newaxis, :] - dot_product
        
        # Avoid division by zero
        eps = 1e-8
        similarity = dot_product / (denominator + eps)
        
        return np.clip(similarity, 0, 1)

    def calculate_similarity(self, X_query_cont, X_query_bin):
        """
        Calculate hybrid similarity.
        """
        # 1. Vector Tanimoto for Continuous Features
        sim_cont = self._calculate_vector_tanimoto(X_query_cont)

        # 2. Standard Tanimoto for Binary Features
        # cdist 'jaccard' returns distance (1 - similarity)
        dist_jaccard = cdist(X_query_bin.astype(bool), self.X_train_bin, 'jaccard')
        sim_bin = 1 - dist_jaccard

        # 3. Weighted Combination
        hybrid_similarity = (self.w_binary * sim_bin) + (self.w_cont * sim_cont)
        
        # Numerical stability
        hybrid_similarity = np.clip(hybrid_similarity, 0, 1)
        if np.isnan(hybrid_similarity).sum() > 0:
            print(f"Warning: NaN values detected in hybrid similarity. Filling with 0.")
            hybrid_similarity = np.nan_to_num(hybrid_similarity, nan=0.0)

        return hybrid_similarity

# ==================== NSG Main Class ====================
class NSG:
    """
    Main Class for Application Domain Evaluation
    """
    
    def __init__(self, df_train, X_train_cont, X_train_bin, yCol):
        self.X_train_cont = X_train_cont
        self.X_train_bin = X_train_bin
        self.df_train = df_train[[yCol]]
        self.yCol = yCol
        
        self._calculate_y_difference_matrices()
        
        # Initialize the Hybrid Tanimoto Calculator
        self.sim_calculator = HybridTanimotoCalculator(
            self.X_train_cont, 
            self.X_train_bin, 
            weight_binary=W_BINARY_PART
        )
        self.dfPSM = None
    
    def _calculate_y_difference_matrices(self):
        y_values = self.df_train[self.yCol].values.reshape(-1, 1)
        self.dfEDM = pd.DataFrame(
            DistanceMetric.get_metric('euclidean').pairwise(y_values),
            index=self.df_train.index, columns=self.df_train.index
        )
        self.dfSDM = pd.DataFrame(
            np.subtract.outer(y_values.flatten(), y_values.flatten()),
            index=self.df_train.index, columns=self.df_train.index
        )
    
    def compute_pairwise_similarity(self):
        similarity = self.sim_calculator.calculate_similarity(self.X_train_cont, self.X_train_bin)
        self.dfPSM = pd.DataFrame(
            similarity,
            index=self.df_train.index,
            columns=self.df_train.index
        )
    
    def compute_local_discontinuity(self, wtFunc=None):
        wtFunc = wtFunc or (lambda x: x)
        dfWt = pd.DataFrame(wtFunc(self.dfPSM.values), index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWPSM = self.dfPSM.multiply(dfWt)
        srWtDg = dfWt.sum(axis=1) - np.diag(dfWt)
        srWtDg.name = f'wtDegree|{self.yCol}'
        dfWtSlopeM = self.dfEDM.values * dfWPSM.values
        eps = 1e-6
        srWtLd = pd.Series(
            dfWtSlopeM.sum(axis=1) / (srWtDg + eps),
            index=self.df_train.index,
            name=f'wtLD|{self.yCol}'
        )
        return self.df_train.join([srWtDg, srWtLd])
    
    def generate_query_similarity_matrix(self, X_test_cont, X_test_bin):
        similarity = self.sim_calculator.calculate_similarity(X_test_cont, X_test_bin)
        dfQTSM = pd.DataFrame(
            similarity,
            index=pd.RangeIndex(start=0, stop=X_test_cont.shape[0], name='TestID'),
            columns=self.df_train.index
        )
        return dfQTSM
    
    def evaluate_application_domain(self, dfQTSM, dfWtLD=None, wtFunc=None, code=''):
        wtFunc = wtFunc or (lambda x: x)
        dfWtLD = dfWtLD or self.compute_local_discontinuity()
        dfSimiWt = pd.DataFrame(wtFunc(dfQTSM.values), index=dfQTSM.index, columns=dfQTSM.columns)
        ld_values = dfWtLD[f'wtLD|{self.yCol}']
        valid_cols = ld_values.dropna().index
        simiDens = dfSimiWt[valid_cols].sum(axis=1)
        simiDens.name = f'simiDensity_a={a}_{code}'
        eps = 1e-6
        wtLD = dfSimiWt[valid_cols].dot(ld_values[valid_cols]) / (simiDens + eps)
        wtLD.name = f'simiWtLD_w_a={a}_{code}'
        return pd.concat([simiDens, wtLD], axis=1)

# ==================== Exponential Weight Function ====================
def exponential_weight(x, a=a, eps=1e-6):
    return np.exp(-a * (1 - x) / (x + eps))

# ==================== Main Execution Flow ====================
if __name__ == "__main__":
    # -------------------- Data Preparation --------------------
    print("Step 1: Loading and preprocessing data...")
    df_train, df_test, X_train_cont, X_train_bin, X_test_cont, X_test_bin = load_and_preprocess_data()
    
    # -------------------- Model Initialization --------------------
    print("Step 2: Initializing NSG evaluator with Hybrid Tanimoto Similarity...")
    nsg = NSG(
        df_train=df_train,
        X_train_cont=X_train_cont,
        X_train_bin=X_train_bin,
        yCol='Target'
    )
    
    # Calculate training set similarity
    print("Step 3: Calculating training set hybrid similarity matrix...")
    nsg.compute_pairwise_similarity()
    
    # -------------------- Application Domain Evaluation --------------------
    print("Step 4: Generating test set hybrid similarity matrix...")
    dfQTSM = nsg.generate_query_similarity_matrix(X_test_cont, X_test_bin)
    
    print("Step 5: Calculating application domain metrics...")
    ad_metrics = nsg.evaluate_application_domain(
        dfQTSM,
        wtFunc=exponential_weight,
        code='exp'
    )
    
    # Merge results and save
    print("Step 6: Saving results...")
    df_test_reset = df_test.reset_index()
    df_test_joined = df_test_reset.join(ad_metrics)
    df_test_joined = df_test_joined.set_index('SampleID')
    
    df_test_joined.to_csv(f'AD_Metrics_a={a}_wBinary={W_BINARY_PART}.csv')
    
    # -------------------- Performance Evaluation --------------------
    print("Step 7: Starting performance evaluation...")
    y_true = df_test['Target']
    y_pred = df_test.get('Predicted')
    
    if y_pred is None:
        print("Error: 'Predicted' column missing in test data.")
    else:
        density_bounds = [round(2 - i * 0.1, 1) for i in range(21)]
        ld_bounds = [i * 0.01 for i in range(0, 150)]
        
        results_n = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_r2 = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_rmse = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_mae = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        
        for density in density_bounds:
            for ld in ld_bounds:
                mask = (
                    (ad_metrics[f'simiDensity_a={a}_exp'] >= density) & 
                    (ad_metrics[f'simiWtLD_w_a={a}_exp'] <= ld)
                )
                valid_samples = df_test.index[mask.values]
                
                if len(valid_samples) > 1:
                    results_n.loc[ld, density] = len(valid_samples)
                    try:
                        current_y_true = y_true.loc[valid_samples]
                        current_y_pred = y_pred.loc[valid_samples]
                        results_r2.loc[ld, density] = metrics.r2_score(current_y_true, current_y_pred)
                        results_rmse.loc[ld, density] = np.sqrt(metrics.mean_squared_error(current_y_true, current_y_pred))
                        results_mae.loc[ld, density] = metrics.mean_absolute_error(current_y_true, current_y_pred)
                    except Exception:
                        results_r2.loc[ld, density] = np.nan
                        results_rmse.loc[ld, density] = np.nan
                        results_mae.loc[ld, density] = np.nan
                else:
                    results_n.loc[ld, density] = len(valid_samples) if len(valid_samples) > 0 else np.nan
                    results_r2.loc[ld, density] = np.nan
                    results_rmse.loc[ld, density] = np.nan
                    results_mae.loc[ld, density] = np.nan
        
        results_n.to_csv(f'result_AD/AD_Validation_n_a={a}_wBinary={W_BINARY_PART}.csv')
        results_r2.to_csv(f'result_AD/AD_Validation_R2_a={a}_wBinary={W_BINARY_PART}.csv')
        results_rmse.to_csv(f'result_AD/AD_Validation_RMSE_a={a}_wBinary={W_BINARY_PART}.csv')
        results_mae.to_csv(f'result_AD/AD_Validation_MAE_a={a}_wBinary={W_BINARY_PART}.csv')
        
    print("All calculations completed! Results have been saved.")
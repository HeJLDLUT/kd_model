# 5_ADFEL.py
"""

ADFEL: Application Domain Characterization Method Based on Feature-Endpoint Topography

## Input
    - Training data: train_data.csv
    - Test data: test_data.csv
    - SHAP weights: shap_values.csv
    - Global parameters:
      - a=5

## Output
    - Application domain metrics:
      - AD_Metrics_a=5.csv
      - Validation results in the result_AD/ directory (n, R2, ERMS, EMA)

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

# Ensure the result folder exists
os.makedirs('result_AD', exist_ok=True)

warnings.filterwarnings('ignore')

# ==================== Global Parameter Configuration ====================
# Exponential weight parameter (controls similarity decay rate)
a = 5

# SHAP file path
SHAP_PATH = 'shap_values.csv'  

# σ calculation parameter
SIGMA_QUANTILE = 0.5  

# ==================== Data Preprocessing Module ====================
def load_and_preprocess_data():
    """
    Load and preprocess training/test data
    Returns:
        X_train: Weighted training feature matrix
        X_test: Weighted test feature matrix
        df_train: Original training DataFrame
        df_test: Original test DataFrame
    """
    # Load raw data
    df_train = pd.read_csv('train_data.csv', index_col='SampleID')
    df_test = pd.read_csv('test_data.csv', index_col='SampleID')
    
    # Define feature columns
    features = [f'Feature_{i}' for i in range(1, 180)]  
    
    # Ensure all features are numeric
    df_train[features] = df_train[features].apply(pd.to_numeric, errors='coerce')
    df_test[features] = df_test[features].apply(pd.to_numeric, errors='coerce')
    
    # Check for NaN values
    nan_train = df_train[features].isnull().sum().sum()
    nan_test = df_test[features].isnull().sum().sum()
    if nan_train > 0 or nan_test > 0:
        print(f"Warning: NaN values detected in the data. Training set NaN count: {nan_train}, Test set NaN count: {nan_test}. Automatically filled with 0.")
        df_train[features] = df_train[features].fillna(0)
        df_test[features] = df_test[features].fillna(0)
    
    return df_train, df_test, features

# ==================== SHAP Weight Processing ====================
def load_and_process_shap_weights():

    try:
        # Read SHAP file
        df_shap = pd.read_csv(SHAP_PATH, index_col=0)
        
        # Select 179 feature columns (SHAP_Feature_1 to SHAP_Feature_179)
        shap_cols = [f'SHAP_Feature_{i}' for i in range(1, 180)]
        df_shap = df_shap[shap_cols]
        
        # Convert to numeric (double-check)
        df_shap = df_shap.apply(pd.to_numeric, errors='coerce')
        nan_shap = df_shap.isnull().sum().sum()
        if nan_shap > 0:
            print(f"Warning: NaN values detected in SHAP values, count: {nan_shap}. Automatically filled with 0.")
            df_shap = df_shap.fillna(0)
        
        # Check if all feature columns are present
        if len(df_shap.columns) != 179:
            print(f"Warning: Missing feature columns in SHAP file. Expected 179 columns, found {len(df_shap.columns)} columns.")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(df_shap.values), axis=0)
        
        # Normalize
        normalized_weights = mean_abs_shap / np.sum(mean_abs_shap)
        print(f"Successfully loaded SHAP weights for {len(normalized_weights)} features.")
        return normalized_weights
        
    except Exception as e:
        print(f"Failed to process SHAP file: {e}\nUsing equal weights as a fallback.")
        return np.ones(179) / 179

# ==================== Core Similarity Calculation Class ====================
class WeightedSimilarityCalculator:
    """
    Weighted Similarity Calculator
    Functions:
    1. Calculate weighted Euclidean distance
    2. Gaussian kernel similarity transformation
    3. Automatic σ parameter calculation
    """
    
    def __init__(self, X_train):
        """
        Initialization
        Parameters:
            X_train: Weighted training feature matrix (n_samples, n_features)
        """
        self.X_train = X_train
        self.sigma = self._calculate_optimal_sigma()
        print(f"Automatically calculated optimal σ parameter: {self.sigma:.4f}")
    
    def _calculate_optimal_sigma(self):
        """
        Calculate optimal σ based on Modified Median Heuristic
        """
        # Random sampling
        sample_size = min(1010, len(self.X_train))
        indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        sampled_data = self.X_train[indices]
        
        # Calculate upper triangular distance matrix
        pairwise_dist = cdist(sampled_data, sampled_data, 'euclidean')
        triu_dist = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
        
        # Calculate corrected median
        median_dist = np.median(triu_dist)
        d = sampled_data.shape[1]  # Feature dimension
        correction = 1 / (np.sqrt(2) * np.power(d, 0.25))  # Dimensionality correction factor
        return max(median_dist * correction, 1e-6)  # Prevent zero value
    
    def calculate_similarity(self, X):
        """
        Calculate Gaussian similarity between input samples and training set
        Parameters:
            X: Test sample feature matrix (n_test, n_features)
        Returns:
            similarity_matrix: Similarity matrix (n_test, n_train)
        """
        # Calculate weighted Euclidean distance
        distances = cdist(X, self.X_train, 'euclidean')
        
        # Check for NaN values in the distance matrix
        nan_dist = np.isnan(distances).sum()
        if nan_dist > 0:
            print(f"Warning: NaN values detected in distance calculation, count: {nan_dist}")
        
        # Gaussian kernel transformation
        similarity = np.exp(-(distances**2) / (2 * self.sigma**2))
        
        # Numerical stability handling
        similarity = np.clip(similarity, 0, 1)
        
        # Check for NaN values in the similarity matrix
        nan_sim = np.isnan(similarity).sum()
        if nan_sim > 0:
            print(f"Warning: NaN values detected in similarity calculation, count: {nan_sim}")
        
        return similarity

# ==================== NSG Main Class ====================
class NSG:
    """
    Main Class for Application Domain Evaluation
    Functions:
    1. Calculate local discontinuity scores
    2. Generate similarity matrices
    3. Evaluate application domain metrics
    """
    
    def __init__(self, df_train, X_train, yCol):
        """
        Initialization
        Parameters:
            df_train: Training DataFrame containing target values
            X_train: Training feature matrix
            yCol: Target column name
        """
        self.X_train = X_train
        self.df_train = df_train[[yCol]]
        self.yCol = yCol
        
        # Initialize difference matrices
        self._calculate_y_difference_matrices()
        
        # Initialize similarity calculator
        self.sim_calculator = WeightedSimilarityCalculator(X_train)
        self.dfPSM = None  # Training set similarity matrix
    
    def _calculate_y_difference_matrices(self):
        """
        Calculate Euclidean and signed distance matrices for the target variable
        """
        y_values = self.df_train[self.yCol].values.reshape(-1, 1)
        
        # Euclidean distance matrix
        self.dfEDM = pd.DataFrame(
            DistanceMetric.get_metric('euclidean').pairwise(y_values),
            index=self.df_train.index,
            columns=self.df_train.index
        )
        
        # Check for NaN values in the Euclidean distance matrix
        nan_edm = self.dfEDM.isnull().sum().sum()
        if nan_edm > 0:
            print(f"Warning: NaN values detected in the Euclidean distance matrix, count: {nan_edm}")
        
        # Signed distance matrix (with direction)
        self.dfSDM = pd.DataFrame(
            np.subtract.outer(y_values.flatten(), y_values.flatten()),
            index=self.df_train.index,
            columns=self.df_train.index
        )
        
        # Check for NaN values in the signed distance matrix
        nan_sdm = self.dfSDM.isnull().sum().sum()
        if nan_sdm > 0:
            print(f"Warning: NaN values detected in the signed distance matrix, count: {nan_sdm}")
    
    def compute_pairwise_similarity(self):
        """
        Calculate pairwise similarity matrix for the training set
        """
        similarity = self.sim_calculator.calculate_similarity(self.X_train)
        self.dfPSM = pd.DataFrame(
            similarity,
            index=self.df_train.index,
            columns=self.df_train.index
        )
        
        # Check for NaN values in the similarity matrix
        nan_psm = self.dfPSM.isnull().sum().sum()
        if nan_psm > 0:
            print(f"Warning: NaN values detected in the training set similarity matrix, count: {nan_psm}")
    
    def compute_local_discontinuity(self, wtFunc=None):
        """
        Calculate weighted local discontinuity scores
        Parameters:
            wtFunc: Weight function (default is exponential weight)
        Returns:
            DataFrame: Results containing wtDegree and wtLD
        """
        wtFunc = wtFunc or (lambda x: x)
        
        # Calculate weight matrix
        dfWt = pd.DataFrame(
            wtFunc(self.dfPSM.values),
            index=self.dfPSM.index,
            columns=self.dfPSM.columns
        )
        
        # Check for NaN values in the weight matrix
        nan_wt = dfWt.isnull().sum().sum()
        if nan_wt > 0:
            print(f"Warning: NaN values detected in the weight matrix, count: {nan_wt}")
        
        # Weighted similarity matrix
        dfWPSM = self.dfPSM.multiply(dfWt)
        
        # Check for NaN values in the weighted similarity matrix
        nan_wpsm = dfWPSM.isnull().sum().sum()
        if nan_wpsm > 0:
            print(f"Warning: NaN values detected in the weighted similarity matrix, count: {nan_wpsm}")
        
        # Calculate weighted degree (excluding diagonal)
        srWtDg = dfWt.sum(axis=1) - np.diag(dfWt)
        srWtDg.name = f'wtDegree|{self.yCol}'
        
        # Check for NaN values in the weighted degree
        nan_wtdg = srWtDg.isnull().sum()
        if nan_wtdg > 0:
            print(f"Warning: NaN values detected in the weighted degree, count: {nan_wtdg}")
        
        # Calculate local discontinuity scores
        dfWtSlopeM = self.dfEDM.values * dfWPSM.values
        dfWtSlopeM_df = pd.DataFrame(dfWtSlopeM, index=self.df_train.index, columns=self.df_train.index)
        
        # Check for NaN values in the local slope matrix
        nan_wtslopem = dfWtSlopeM_df.isnull().sum().sum()
        if nan_wtslopem > 0:
            print(f"Warning: NaN values detected in the local slope matrix, count: {nan_wtslopem}")
        
        eps = 1e-6
        srWtLd = pd.Series(
            dfWtSlopeM.sum(axis=1) / (srWtDg + eps),
            index=self.df_train.index,
            name=f'wtLD|{self.yCol}'
        )
        
        # Check for NaN values in the local discontinuity scores
        nan_wtld = srWtLd.isnull().sum()
        if nan_wtld > 0:
            print(f"Warning: NaN values detected in the local discontinuity scores, count: {nan_wtld}")
        
        return self.df_train.join([srWtDg, srWtLd])
    
    def generate_query_similarity_matrix(self, X_test):
        """
        Generate query-training similarity matrix
        Parameters:
            X_test: Test feature matrix
        Returns:
            DataFrame: Similarity matrix (n_test, n_train)
        """
        similarity = self.sim_calculator.calculate_similarity(X_test)
        dfQTSM = pd.DataFrame(
            similarity,
            index=pd.RangeIndex(start=0, stop=X_test.shape[0], name='TestID'),
            columns=self.df_train.index
        )
        
        # Check for NaN values in the query-training similarity matrix
        nan_qtsm = dfQTSM.isnull().sum().sum()
        if nan_qtsm > 0:
            print(f"Warning: NaN values detected in the query-training similarity matrix, count: {nan_qtsm}")
        
        return dfQTSM
    
    def evaluate_application_domain(self, dfQTSM, dfWtLD=None, wtFunc=None, code=''):
        """
        Evaluate application domain metrics
        Parameters:
            dfQTSM: Query-training similarity matrix
            dfWtLD: Precomputed local discontinuity scores
            wtFunc: Weight function
        Returns:
            DataFrame: Results containing similarity density and weighted LD
        """
        wtFunc = wtFunc or (lambda x: x)
        dfWtLD = dfWtLD or self.compute_local_discontinuity()
        
        # Calculate similarity weights
        dfSimiWt = pd.DataFrame(
            wtFunc(dfQTSM.values),
            index=dfQTSM.index,
            columns=dfQTSM.columns
        )
        
        # Check for NaN values in the similarity weight matrix
        nan_simiw = dfSimiWt.isnull().sum().sum()
        if nan_simiw > 0:
            print(f"Warning: NaN values detected in the similarity weight matrix, count: {nan_simiw}")
        
        # Get valid LD values (non-NA)
        ld_values = dfWtLD[f'wtLD|{self.yCol}']
        valid_cols = ld_values.dropna().index
        
        # Calculate similarity density
        simiDens = dfSimiWt[valid_cols].sum(axis=1)
        simiDens.name = f'simiDensity_a={a}_{code}'
        
        # Check for NaN values in the similarity density
        nan_simidens = simiDens.isnull().sum()
        if nan_simidens > 0:
            print(f"Warning: NaN values detected in the similarity density, count: {nan_simidens}")
        
        # Calculate weighted LD
        eps = 1e-6
        wtLD = dfSimiWt[valid_cols].dot(ld_values[valid_cols]) / (simiDens + eps)
        wtLD.name = f'simiWtLD_w_a={a}_{code}'
        
        # Check for NaN values in the weighted LD
        nan_wtld_final = wtLD.isnull().sum()
        if nan_wtld_final > 0:
            print(f"Warning: NaN values detected in the weighted LD, count: {nan_wtld_final}")
        
        return pd.concat([simiDens, wtLD], axis=1)

# ==================== Exponential Weight Function ====================
def exponential_weight(x, a=a, eps=1e-6):
    """
    Exponential weight function
    Parameters:
        x: Similarity value (0-1)
        a: Decay coefficient
        eps: Numerical stability term
    Returns:
        Weight value
    """
    return np.exp(-a * (1 - x) / (x + eps))

# ==================== Main Execution Flow ====================
if __name__ == "__main__":
    # -------------------- Data Preparation --------------------
    print("Step 1: Loading and preprocessing data...")
    df_train, df_test, features = load_and_preprocess_data()
    
    # Load SHAP weights
    #print("Step 2: Loading SHAP weights...")
    #shap_weights = load_and_process_shap_weights()
    
    # Feature weighting
    print("Step 3: Applying SHAP weights to features...")
    #X_train = df_train[features].values * np.sqrt(shap_weights)
    #X_test = df_test[features].values * np.sqrt(shap_weights)

    X_train = df_train[features].values
    X_test = df_test[features].values
    
    # -------------------- Model Initialization --------------------
    print("Step 4: Initializing NSG evaluator...")
    nsg = NSG(
        df_train=df_train,
        X_train=X_train,
        yCol='Target'  # Assuming the target column name is 'Target'
    )
    
    # Calculate training set similarity
    print("Step 5: Calculating training set similarity matrix...")
    nsg.compute_pairwise_similarity()
    
    # -------------------- Application Domain Evaluation --------------------
    print("Step 6: Generating test set similarity matrix...")
    dfQTSM = nsg.generate_query_similarity_matrix(X_test)
    
    print("Step 7: Calculating application domain metrics...")
    ad_metrics = nsg.evaluate_application_domain(
        dfQTSM,
        wtFunc=exponential_weight,
        code='exp'
    )
    
    # Merge results and save
    print("Step 8: Saving results...")
    df_test_joined = df_test.join(ad_metrics)
    
    # Check for NaN values in the merged data
    nan_joined = df_test_joined.isnull().sum().sum()
    if nan_joined > 0:
        print(f"Warning: NaN values detected in the merged data, count: {nan_joined}")
    
    df_test_joined.to_csv(f'AD_Metrics_a={a}.csv')
    
    # -------------------- Performance Evaluation (Retain Original Logic) --------------------
    print("Step 9: Starting performance evaluation...")
    y_true = df_test['Target']
    y_pred = df_test.get('Predicted')  # Use get to avoid errors if the column is missing
    
    if y_pred is None:
        print("Error: The test set is missing the 'Predicted' column. Please ensure predictions are available.")
    else:
        # Parameter range settings (retain original configuration)
        density_bounds = [round(2 - i * 0.1, 1) for i in range(21)]
        ld_bounds = [i * 0.01 for i in range(0, 150)]
        
        # Initialize result containers
        results_n = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_r2 = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_rmse = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_mae = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        
        # Grid search evaluation
        for density in density_bounds:
            for ld in ld_bounds:
                # Filter samples meeting AD conditions
                mask = (
                    (ad_metrics[f'simiDensity_a={a}_exp'] >= density) & 
                    (ad_metrics[f'simiWtLD_w_a={a}_exp'] <= ld)
                )
                
                valid_samples = df_test.index[mask]
                
                if len(valid_samples) > 0:
                    # Record sample count
                    results_n.loc[ld, density] = len(valid_samples)
                    
                    # Calculate metrics
                    try:
                        current_y_true = y_true.loc[valid_samples]
                        current_y_pred = y_pred.loc[valid_samples]
                        
                        # Check for NaN values in y_true and y_pred
                        nan_true = current_y_true.isnull().sum()
                        nan_pred = current_y_pred.isnull().sum()
                        if nan_true > 0 or nan_pred > 0:
                            print(f"Warning: NaN values detected in evaluation samples, density={density}, ld={ld}. y_true NaN count: {nan_true}, y_pred NaN count: {nan_pred}")
                        
                        results_r2.loc[ld, density] = metrics.r2_score(
                            current_y_true, current_y_pred)
                        results_rmse.loc[ld, density] = np.sqrt(
                            metrics.mean_squared_error(current_y_true, current_y_pred))
                        results_mae.loc[ld, density] = metrics.mean_absolute_error(
                            current_y_true, current_y_pred)
                    except Exception as e:
                        print(f"Evaluation failed for density={density}, ld={ld}: {str(e)}")
                        results_r2.loc[ld, density] = np.nan
                        results_rmse.loc[ld, density] = np.nan
                        results_mae.loc[ld, density] = np.nan
                else:
                    results_r2.loc[ld, density] = np.nan
                    results_rmse.loc[ld, density] = np.nan
                    results_mae.loc[ld, density] = np.nan
        
        # Check for NaN values in evaluation results
        nan_results_n = results_n.isnull().sum().sum()
        nan_results_r2 = results_r2.isnull().sum().sum()
        nan_results_rmse = results_rmse.isnull().sum().sum()
        nan_results_mae = results_mae.isnull().sum().sum()
        
        if nan_results_n > 0:
            print(f"Warning: NaN values detected in the 'results_n' container, count: {nan_results_n}")
        if nan_results_r2 > 0:
            print(f"Warning: NaN values detected in the 'results_r2' container, count: {nan_results_r2}")
        if nan_results_rmse > 0:
            print(f"Warning: NaN values detected in the 'results_rmse' container, count: {nan_results_rmse}")
        if nan_results_mae > 0:
            print(f"Warning: NaN values detected in the 'results_mae' container, count: {nan_results_mae}")

        # Save evaluation results
        results_n.to_csv(f'result_AD/AD_Validation_n_a={a}.csv')
        results_r2.to_csv(f'result_AD/AD_Validation_R2_a={a}.csv')
        results_rmse.to_csv(f'result_AD/AD_Validation_RMSE_a={a}.csv')
        results_mae.to_csv(f'result_AD/AD_Validation_MAE_a={a}.csv')
        
        print("All calculations completed! Results have been saved to CSV files.")

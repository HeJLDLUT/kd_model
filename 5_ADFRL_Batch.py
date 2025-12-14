# 5-2_ADFRL_Batch.py
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cdist
import warnings
import os
import glob

# Configuration
os.makedirs('result_AD', exist_ok=True)
warnings.filterwarnings('ignore')

# Global Parameters
a = 25
# Define the list of weights to investigate (Ablation Study)
WEIGHT_LIST = [0.0, 0.2, 0.4, 0.6,0.8, 1.0]

def load_and_preprocess_data():
    result_dir = "results-xgb"
    train_files = glob.glob(os.path.join(result_dir, "*_best_train_data.csv"))
    if not train_files:
        raise FileNotFoundError(f"No training data found in {result_dir}")
    
    latest_train_file = max(train_files, key=os.path.getmtime)
    latest_test_file = latest_train_file.replace("_best_train_data.csv", "_best_validation_data.csv")
    
    if not os.path.exists(latest_test_file):
        raise FileNotFoundError(f"Validation file not found: {latest_test_file}")

    print(f"Loading Train: {latest_train_file}")
    print(f"Loading Test:  {latest_test_file}")

    df_train = pd.read_csv(latest_train_file, index_col='SampleID')
    df_test = pd.read_csv(latest_test_file, index_col='SampleID')
    
    continuous_features = [f'Feature_{i}' for i in range(1, 13)]
    binary_features = [f'Feature_{i}' for i in range(13, 180)]
    all_features = continuous_features + binary_features
    
    df_train[all_features] = df_train[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_test[all_features] = df_test[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_train_cont = df_train[continuous_features].values
    X_train_bin = df_train[binary_features].values
    X_test_cont = df_test[continuous_features].values
    X_test_bin = df_test[binary_features].values

    return df_train, df_test, X_train_cont, X_train_bin, X_test_cont, X_test_bin

class HybridTanimotoCalculator:
    def __init__(self, X_train_cont, X_train_bin, weight_binary):
        self.X_train_cont = X_train_cont
        self.X_train_bin = X_train_bin.astype(bool)
        self.w_binary = weight_binary
        self.w_cont = 1 - weight_binary
        self.train_cont_norm_sq = np.sum(self.X_train_cont ** 2, axis=1)

    def _calculate_vector_tanimoto(self, X_query):
        dot_product = np.dot(X_query, self.X_train_cont.T)
        query_norm_sq = np.sum(X_query ** 2, axis=1)
        denominator = query_norm_sq[:, np.newaxis] + self.train_cont_norm_sq[np.newaxis, :] - dot_product
        eps = 1e-8
        similarity = dot_product / (denominator + eps)
        return np.clip(similarity, 0, 1)

    def calculate_similarity(self, X_query_cont, X_query_bin):
        if self.w_cont > 0:
            sim_cont = self._calculate_vector_tanimoto(X_query_cont)
        else:
            sim_cont = np.zeros((X_query_cont.shape[0], self.X_train_cont.shape[0]))

        if self.w_binary > 0:
            dist_jaccard = cdist(X_query_bin.astype(bool), self.X_train_bin, 'jaccard')
            sim_bin = 1 - dist_jaccard
        else:
            sim_bin = np.zeros((X_query_bin.shape[0], self.X_train_bin.shape[0]))

        hybrid_similarity = (self.w_binary * sim_bin) + (self.w_cont * sim_cont)
        hybrid_similarity = np.clip(hybrid_similarity, 0, 1)
        return np.nan_to_num(hybrid_similarity, nan=0.0)

class NSG:
    def __init__(self, df_train, X_train_cont, X_train_bin, yCol, weight_binary):
        self.df_train = df_train[[yCol]]
        self.yCol = yCol
        self.X_train_cont = X_train_cont
        self.X_train_bin = X_train_bin
        
        y_values = self.df_train[self.yCol].values.reshape(-1, 1)
        self.dfEDM = pd.DataFrame(
            DistanceMetric.get_metric('euclidean').pairwise(y_values),
            index=self.df_train.index, columns=self.df_train.index
        )
        
        self.sim_calculator = HybridTanimotoCalculator(X_train_cont, X_train_bin, weight_binary)
        self.dfPSM = None
    
    def compute_pairwise_similarity(self):
        similarity = self.sim_calculator.calculate_similarity(self.X_train_cont, self.X_train_bin)
        self.dfPSM = pd.DataFrame(similarity, index=self.df_train.index, columns=self.df_train.index)
    
    def compute_local_discontinuity(self, wtFunc):
        dfWt = pd.DataFrame(wtFunc(self.dfPSM.values), index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWPSM = self.dfPSM.multiply(dfWt)
        srWtDg = dfWt.sum(axis=1) - np.diag(dfWt)
        dfWtSlopeM = self.dfEDM.values * dfWPSM.values
        eps = 1e-6
        srWtLd = pd.Series(dfWtSlopeM.sum(axis=1) / (srWtDg + eps), index=self.df_train.index, name=f'wtLD|{self.yCol}')
        return self.df_train.join([srWtDg, srWtLd])
    
    def generate_query_similarity_matrix(self, X_test_cont, X_test_bin):
        similarity = self.sim_calculator.calculate_similarity(X_test_cont, X_test_bin)
        return pd.DataFrame(similarity, index=pd.RangeIndex(X_test_cont.shape[0], name='TestID'), columns=self.df_train.index)
    
    def evaluate_application_domain(self, dfQTSM, dfWtLD, wtFunc):
        dfSimiWt = pd.DataFrame(wtFunc(dfQTSM.values), index=dfQTSM.index, columns=dfQTSM.columns)
        ld_values = dfWtLD[f'wtLD|{self.yCol}']
        valid_cols = ld_values.dropna().index
        simiDens = dfSimiWt[valid_cols].sum(axis=1)
        simiDens.name = 'simiDensity'
        eps = 1e-6
        wtLD = dfSimiWt[valid_cols].dot(ld_values[valid_cols]) / (simiDens + eps)
        wtLD.name = 'simiWtLD'
        return pd.concat([simiDens, wtLD], axis=1)

def exponential_weight(x, a=a, eps=1e-6):
    return np.exp(-a * (1 - x) / (x + eps))

if __name__ == "__main__":
    # 1. Load Data (Once)
    df_train, df_test, X_train_cont, X_train_bin, X_test_cont, X_test_bin = load_and_preprocess_data()
    y_true = df_test['Target']
    y_pred = df_test.get('Predicted')
    
    if y_pred is None:
        raise ValueError("Error: 'Predicted' column missing in test data.")

    # 2. Loop through weights
    for w in WEIGHT_LIST:
        print(f"\n=== Processing Weight Binary = {w} ===")
        
        # Initialize NSG
        nsg = NSG(df_train, X_train_cont, X_train_bin, 'Target', weight_binary=w)
        nsg.compute_pairwise_similarity()
        
        # Compute Training LD
        dfWtLD = nsg.compute_local_discontinuity(exponential_weight)
        
        # Compute Test Similarity
        dfQTSM = nsg.generate_query_similarity_matrix(X_test_cont, X_test_bin)
        
        # Evaluate AD
        ad_metrics = nsg.evaluate_application_domain(dfQTSM, dfWtLD, exponential_weight)
        
        # Save raw metrics
        df_res = df_test.copy()
        df_res = df_res.reset_index().join(ad_metrics).set_index('SampleID')
        df_res.to_csv(f'AD_Metrics_a={a}_wBinary={w}.csv')
        
        # Validation Grid
        density_bounds = [round(2 - i * 0.1, 1) for i in range(21)]
        ld_bounds = [i * 0.01 for i in range(0, 150)]
        
        results_n = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        results_rmse = pd.DataFrame(index=ld_bounds, columns=density_bounds)
        
        print("  > Calculating validation grid...")
        for density in density_bounds:
            for ld in ld_bounds:
                mask = (ad_metrics['simiDensity'] >= density) & (ad_metrics['simiWtLD'] <= ld)
                valid_samples = df_test.index[mask.values]
                
                count = len(valid_samples)
                results_n.loc[ld, density] = count
                
                if count > 1:
                    rmse = np.sqrt(metrics.mean_squared_error(y_true.loc[valid_samples], y_pred.loc[valid_samples]))
                    results_rmse.loc[ld, density] = rmse
                else:
                    results_rmse.loc[ld, density] = np.nan
        
        # Save Validation Matrices
        results_n.to_csv(f'result_AD/AD_Validation_n_a={a}_wBinary={w}.csv')
        results_rmse.to_csv(f'result_AD/AD_Validation_RMSE_a={a}_wBinary={w}.csv')
        print(f"  > Saved results for w={w}")

    print("\nAll weights processed successfully.")
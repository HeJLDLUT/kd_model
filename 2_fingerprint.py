# 2_fingerprint.py
"""
## Function Overview
Batch generation of various molecular fingerprint features (ECFP4/ECFP6/MACCS/AFP/PubChem).

## Input
    - Preprocessed feature file: `processed_features.csv`
    - Fingerprint configuration parameters:
      - ECFP4/ECFP6: Radius (2/3), fingerprint length (512)
      - MACCS: Fixed 167 dimensions
      - AFP: 512-dimensional Avalon fingerprint
      - PubChem: 881-dimensional Morgan fingerprint

## Output
    - Multiple fingerprint files (saved separately by type):
      - ECFP4_512.csv
      - ECFP6_512.csv 
      - MACCS.csv
      - AFP_512.csv
      - PubChemFP.csv

@author: HeJL_DLUT
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
import deepchem as dc
from rdkit import RDLogger

# Disable RDKit warning logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Configuration parameters
INPUT_CSV = "processed_features.csv"
OUTPUT_FILES = {
    "ECFP4": ("ECFP4_512.csv", 512),
    "ECFP6": ("ECFP6_512.csv", 512),
    "MACCS": ("MACCS.csv", 167),
    "AFP": ("AFP_512.csv", 512),
    "PubChem": ("PubChemFP.csv", 881),
}
FINGERPRINT_CONFIG = {
    "ECFP4": {"radius": 2, "size": 512, "chiral": False, "bonds": True, "features": False},
    "ECFP6": {"radius": 3, "size": 512, "chiral": False, "bonds": True, "features": False},
}

def generate_fingerprints(smiles):
    """
    Generate different types of molecular fingerprints based on SMILES.
    """
    fingerprints = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # For invalid SMILES, generate zero-filled fingerprints
        fingerprints["ECFP4"] = [0] * FINGERPRINT_CONFIG["ECFP4"]["size"]
        fingerprints["ECFP6"] = [0] * FINGERPRINT_CONFIG["ECFP6"]["size"]
        fingerprints["MACCS"] = [0] * 167
        fingerprints["AFP"] = [0] * 512
        fingerprints["PubChem"] = [0] * 881
        return fingerprints

    # ECFP4
    ecfp4_fp = dc.feat.CircularFingerprint(**FINGERPRINT_CONFIG["ECFP4"])([mol])[0]
    fingerprints["ECFP4"] = ecfp4_fp.tolist()

    # ECFP6
    ecfp6_fp = dc.feat.CircularFingerprint(**FINGERPRINT_CONFIG["ECFP6"])([mol])[0]
    fingerprints["ECFP6"] = ecfp6_fp.tolist()

    # MACCS
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)
    fingerprints["MACCS"] = maccs_array.tolist()

    # AFP
    afp_fp = pyAvalonTools.GetAvalonFP(mol, 512)
    afp_array = np.zeros((512,), dtype=int)
    DataStructs.ConvertToNumpyArray(afp_fp, afp_array)
    fingerprints["AFP"] = afp_array.tolist()

    # PubChem
    pubchem_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
    pubchem_array = np.zeros((881,), dtype=int)
    DataStructs.ConvertToNumpyArray(pubchem_fp, pubchem_array)
    fingerprints["PubChem"] = pubchem_array.tolist()

    return fingerprints

def main():
    # Read SMILES data
    df = pd.read_csv(INPUT_CSV)
    smiles_list = df["SMILES"].tolist()
    
    # Initialize fingerprint dictionary
    fingerprint_data = {fp_type: [] for fp_type in OUTPUT_FILES.keys()}

    # Process SMILES one by one and generate fingerprints
    for i, smiles in enumerate(smiles_list, 1):
        fingerprints = generate_fingerprints(smiles)
        for fp_type in OUTPUT_FILES.keys():
            fingerprint_data[fp_type].append(fingerprints[fp_type])
        if i % 100 == 0 or i == len(smiles_list):
            print(f"Processed {i}/{len(smiles_list)} SMILES")

    # Save fingerprints to CSV files
    for fp_type, (filename, vec_size) in OUTPUT_FILES.items():
        df_fp = pd.DataFrame(fingerprint_data[fp_type], 
                            columns=[f"feature_{i}" for i in range(vec_size)])
        # Write without index, do not save compound_index column
        df_fp.to_csv(filename, index=False)
        print(f"Saved {fp_type} to {filename}")

if __name__ == "__main__":
    main()

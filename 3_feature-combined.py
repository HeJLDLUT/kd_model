# 3_feature-combined.py
"""
## Function Overview
Integrate SMILES sequences, molecular graph structures, experimental conditions, and fingerprint features to generate multimodal feature combinations.

## Input
    - Preprocessed features: `processed_features.csv`
    - Fingerprint files: `AFP_512.csv`/`ECFP4_512.csv`/`ECFP6_512.csv`/`MACCS.csv`/`PubChemFP.csv`

## Output
    - Multimodal feature sets (*.npy):
      - **Basic features**: `feature_smiles.npy` (serialized SMILES)
      - **Graph structure features**: `feature_moleculargraph.npy` (atom/edge features)
      - **Combined features**:
      - **Fingerprint fusion features**: 5 types of fingerprints + condition features

@author: HeJL_DLUT
"""

import torch
from rdkit import Chem
import pandas as pd
import numpy as np
import os

# Create output folder
os.makedirs('input_features', exist_ok=True)


# Read the experimental condition feature file
df_conditions = pd.read_csv('processed_features.csv', encoding='gbk')

# Extract condition features, including numerical features and one-hot encoded categorical features
condition_features = df_conditions.iloc[:, 1:].values  

# Print rows containing non-numeric data
for i, row in enumerate(condition_features):
    try:
        row = row.astype(np.float32)
    except ValueError as e:
        print(f"Row {i} contains non-numeric data: {row}")
        print(f"Error: {e}")

# Check data types and convert to float
# First, replace all non-numeric data with NaN
condition_features = pd.DataFrame(condition_features).apply(pd.to_numeric, errors='coerce').values

# Print the number of rows containing NaN
print("Number of rows with NaN values:", np.isnan(condition_features).sum())

# Remove rows containing NaN
condition_features = condition_features[~np.isnan(condition_features).any(axis=1)]

# Convert to torch.tensor
condition_features = torch.tensor(condition_features, dtype=torch.float)

# Print data type and shape
print("Condition features dtype:", condition_features.dtype)
print("Condition features shape:", condition_features.shape)

# Extract SMILES and target values
smiles_list = df_conditions.iloc[:, 0].tolist()  # The first column is the SMILES code

# Clean the SMILES list to ensure all data are strings
smiles_list = [str(smiles) for smiles in smiles_list]

# 1. Convert SMILES strings to sequences

# First, create a vocabulary for the SMILES character set
def build_vocab(smiles_list):
    vocab = set()
    for smiles in smiles_list:
        vocab.update(smiles) 
    vocab = sorted(vocab)  
    # Add an extra character for padding
    vocab.append('E') 
    return {char: idx for idx, char in enumerate(vocab)}, vocab

# Use the vocabulary to convert SMILES to sequences
def smiles_to_seq(smiles, vocab):
    return [vocab[char] for char in smiles]

# Check the validity of SMILES using RDKit
def check_smiles_validity(smiles_list):
    valid_indices = [i for i, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(smiles) is not None]
    return [smiles_list[i] for i in valid_indices]

# Check and clean invalid SMILES strings
smiles_list = check_smiles_validity(smiles_list)

# Build the vocabulary
vocab, vocab_list = build_vocab(smiles_list)

# Convert SMILES strings to numerical sequences
smiles_seq = [smiles_to_seq(smiles, vocab) for smiles in smiles_list]

# Find the length of the longest sequence
max_seq_length = max(len(seq) for seq in smiles_seq)

# Pad sequences
def pad_sequences(sequences, max_length, padding_value):
    padded_sequences = []
    for seq in sequences:
        seq += [padding_value] * (max_length - len(seq)) 
        padded_sequences.append(seq)
    return padded_sequences

# Use the 'E' character index in the vocabulary for padding
padding_value = vocab['E']

# Pad the SMILES sequences
padded_smiles_seq = pad_sequences(smiles_seq, max_seq_length, padding_value)
padded_smiles_seq = torch.tensor(padded_smiles_seq, dtype=torch.long)

# Print shape
print("Padded SMILES sequence shape:", padded_smiles_seq.shape)

# 2. Convert SMILES codes to graph data and encode their node and edge features as sequences
# Generate atom features
def atom_features(atom):
    """Return a list of atom properties."""
    return [
        atom.GetAtomicNum(), 
        atom.GetTotalDegree(), 
        atom.GetFormalCharge(), 
        atom.GetNumRadicalElectrons(), 
        atom.GetHybridization().real, 
        atom.GetIsAromatic(), 
        atom.GetTotalNumHs(), 
        atom.HasProp('_ChiralityPossible'), 
        atom.GetChiralTag().real if atom.HasProp('_ChiralityPossible') else 0, 
    ]

# Generate bond features
def bond_features(bond):
    """Return a list of bond properties."""
    bt = bond.GetBondType()
    return [
        bt.real, 
        bond.GetIsConjugated(),
        bond.IsInRing(), 
        bond.GetStereo().real, 
    ]

# Convert graph data to sequences
def graph_to_seq(smiles, atom_feat_fn=atom_features, bond_feat_fn=bond_features):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # Get atom features
    atoms = mol.GetAtoms()
    atom_features_list = [atom_feat_fn(atom) for atom in atoms]

    # Create an array to store edge features
    edge_indices = []
    edge_features_list = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Add edge indices
        edge_indices += [(start, end), (end, start)]
        # Add edge features
        edge_features_list += [bond_feat_fn(bond), bond_feat_fn(bond)]
    
    # Convert to tensors for model input
    node_features_tensor = torch.tensor(atom_features_list, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features_tensor = torch.tensor(edge_features_list, dtype=torch.float)

    return node_features_tensor, edge_index_tensor, edge_features_tensor

# Allocate placeholders for graph data node and edge features
graph_node_features = []
graph_edge_indices = []
graph_edge_features = []

# Loop through the SMILES sequences for conversion
for smiles in smiles_list:
    node_feats, edge_idxs, edge_feats = graph_to_seq(smiles)
    if node_feats is not None:
        graph_node_features.append(node_feats)
        graph_edge_indices.append(edge_idxs)
        graph_edge_features.append(edge_feats)

# Calculate the maximum number of atoms across all molecules
max_atoms = max(len(node_feats) for node_feats in graph_node_features)

# Define a function to pad node features
def pad_node_features(node_features, max_atoms, n_atom_features):
    """Fill the node feature list for each molecule to make the length uniform."""
    padded_node_features = []
    for features_list in node_features:
        # Number of padding entries to be added
        padding_length = max_atoms - len(features_list)
        # Create a single tensor of zeros for padding with the shape (padding_length, n_atom_features)
        if padding_length > 0:
            padding = torch.zeros(padding_length, n_atom_features)
            padded_features_list = torch.cat((features_list, padding), dim=0)
        else:
            padded_features_list = features_list
        # Append the current padded features list to the output list
        padded_node_features.append(padded_features_list)
    return torch.stack(padded_node_features)

# Now pad the node features
n_atom_features = len(graph_node_features[0][0])  
padded_node_features = pad_node_features(graph_node_features, max_atoms, n_atom_features)

reshaped_node_features = torch.flatten(padded_node_features, start_dim=1)

# Print shape
print("Reshaped node features shape:", reshaped_node_features.shape)

# Weight factors
weight_padded_smiles_seq = 1
weight_reshaped_node_features = 1
weight_condition_features = 1

# Adjust feature values by multiplying with weight factors
weighted_padded_smiles_seq = padded_smiles_seq * weight_padded_smiles_seq
weighted_reshaped_node_features = reshaped_node_features * weight_reshaped_node_features
weighted_condition_features = condition_features * weight_condition_features

# Ensure the first dimension of both is the same, i.e., the number of data points
print("Padded SMILES sequence shape:", padded_smiles_seq.shape)
print("Reshaped node features shape:", reshaped_node_features.shape)
print("Condition features shape:", condition_features.shape)

feature_smiles = padded_smiles_seq
feature_moleculargraph = reshaped_node_features
feature_smiles_condition = torch.cat((padded_smiles_seq, condition_features), dim=1)
feature_moleculargraph_condition = torch.cat((reshaped_node_features, condition_features), dim=1)

# Save X_combined as Numpy array files
np.save('input_features/feature_smiles.npy', feature_smiles)
np.save('input_features/feature_moleculargraph.npy', feature_moleculargraph)
np.save('input_features/feature_smiles_condition.npy', feature_smiles_condition)
np.save('input_features/feature_moleculargraph_condition.npy', feature_moleculargraph_condition)

# Read fingerprint files
AFP_512 = pd.read_csv('AFP_512.csv', encoding='utf-8')
ECFP4_512 = pd.read_csv('ECFP4_512.csv', encoding='utf-8')
ECFP6_512 = pd.read_csv('ECFP6_512.csv', encoding='utf-8')
MACCS = pd.read_csv('MACCS.csv', encoding='utf-8')

# Process AFP_512 data
AFP_512 = AFP_512.apply(pd.to_numeric, errors='coerce')
AFP_512 = AFP_512.fillna(0)
AFP_512_features = torch.tensor(AFP_512.values, dtype=torch.float)

# Process ECFP4_512 data
# Exclude non-numeric columns
ECFP4_512 = ECFP4_512.apply(pd.to_numeric, errors='coerce')
ECFP4_512 = ECFP4_512.fillna(0)
ECFP4_512_features = torch.tensor(ECFP4_512.values, dtype=torch.float)

# Process ECFP6_512 data
ECFP6_512 = ECFP6_512.apply(pd.to_numeric, errors='coerce')
ECFP6_512 = ECFP6_512.fillna(0)
ECFP6_512_features = torch.tensor(ECFP6_512.values, dtype=torch.float)

# Process MACCS data
MACCS = MACCS.apply(pd.to_numeric, errors='coerce')
MACCS = MACCS.fillna(0)
MACCS_features = torch.tensor(MACCS.values, dtype=torch.float)

# Concatenate features
X_combined_F_AFP_512 = torch.cat((AFP_512_features,))
X_combined_F_ECFP4_512 = torch.cat((ECFP4_512_features,))
X_combined_F_ECFP6_512 = torch.cat((ECFP6_512_features,))
X_combined_F_MACCS = torch.cat((MACCS_features,))

X_combined_F_AFP_512_C = torch.cat((condition_features, AFP_512_features), dim=1)
X_combined_F_ECFP4_512_C = torch.cat((condition_features, ECFP4_512_features), dim=1)
X_combined_F_ECFP6_512_C = torch.cat((condition_features, ECFP6_512_features), dim=1)
X_combined_F_MACCS_C = torch.cat((condition_features, MACCS_features), dim=1)

# Save new feature tensors as Numpy array files
np.save('input_features/AFP_512.npy', X_combined_F_AFP_512)
np.save('input_features/ECFP4_512.npy', X_combined_F_ECFP4_512)
np.save('input_features/ECFP6_512.npy', X_combined_F_ECFP6_512)
np.save('input_features/MACCS.npy', X_combined_F_MACCS)

np.save('input_features/AFP_512_condition.npy', X_combined_F_AFP_512_C)
np.save('input_features/ECFP4_512_condition.npy', X_combined_F_ECFP4_512_C)
np.save('input_features/ECFP6_512_condition.npy', X_combined_F_ECFP6_512_C)
np.save('input_features/MACCS_condition.npy', X_combined_F_MACCS_C)

# Read fingerprint file
PubChemFP = pd.read_csv('PubChemFP.csv', encoding='utf-8')

# Convert to tensor
PubChemFP_features = torch.tensor(PubChemFP.values, dtype=torch.float)

# Concatenate features
X_combined_F_PubChemFP = torch.cat((PubChemFP_features,))
X_combined_F_PubChemFP_C = torch.cat((condition_features, PubChemFP_features), dim=1)

# Save new feature tensors as Numpy array files
np.save('input_features/PubChemFP.npy', X_combined_F_PubChemFP)
np.save('input_features/PubChemFP_condition.npy', X_combined_F_PubChemFP_C)

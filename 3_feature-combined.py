# 3_feature-combined.py
"""
...
@author: HeJL_DLUT
"""

import torch
from rdkit import Chem
import pandas as pd
import numpy as np
import os

os.makedirs('input_features', exist_ok=True)

df_conditions = pd.read_csv('processed_features.csv', encoding='utf-8')
condition_features = df_conditions.iloc[:, 1:].values

for i, row in enumerate(condition_features):
    try:
        row = row.astype(np.float32)
    except ValueError as e:
        print(f"Row {i} contains non-numeric data: {row}")
        print(f"Error: {e}")

condition_features = pd.DataFrame(condition_features).apply(pd.to_numeric, errors='coerce').values
print("Number of rows with NaN values:", np.isnan(condition_features).sum())
condition_features = condition_features[~np.isnan(condition_features).any(axis=1)]
condition_features = torch.tensor(condition_features, dtype=torch.float)
print("Initial Condition features shape:", condition_features.shape)

smiles_list = df_conditions.iloc[:, 0].tolist()
smiles_list = [str(smiles) for smiles in smiles_list]


invalid_smiles_info = []
for i, smiles in enumerate(smiles_list):
    if Chem.MolFromSmiles(smiles) is None:

        invalid_smiles_info.append((i, smiles))

# Print the findings in a user-friendly format.
if invalid_smiles_info:
    print("\n--- Invalid SMILES Detected ---")
    print(f"Found {len(invalid_smiles_info)} invalid SMILES which will be filtered out:")
    for index, smiles in invalid_smiles_info:

        print(f"  - Original CSV Row {index + 2} (Index {index}): '{smiles}'")
    print("---------------------------------\n")
else:
    print("\n--- SMILES Validity Check ---")
    print("All SMILES were found to be valid.")
    print("-----------------------------\n")

valid_indices = [i for i, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(smiles) is not None]


smiles_list = [smiles_list[i] for i in valid_indices]


condition_features = condition_features[valid_indices]


print("Condition features shape after filtering:", condition_features.shape)

def build_vocab(smiles_list):
    vocab = set()
    for smiles in smiles_list:
        vocab.update(smiles) 
    vocab = sorted(vocab)  
    vocab.append('E') 
    return {char: idx for idx, char in enumerate(vocab)}, vocab

def smiles_to_seq(smiles, vocab):
    return [vocab[char] for char in smiles]



vocab, vocab_list = build_vocab(smiles_list)
smiles_seq = [smiles_to_seq(smiles, vocab) for smiles in smiles_list]
max_seq_length = max(len(seq) for seq in smiles_seq)

def pad_sequences(sequences, max_length, padding_value):
    padded_sequences = []
    for seq in sequences:
        seq += [padding_value] * (max_length - len(seq)) 
        padded_sequences.append(seq)
    return padded_sequences

padding_value = vocab['E']
padded_smiles_seq = pad_sequences(smiles_seq, max_seq_length, padding_value)
padded_smiles_seq = torch.tensor(padded_smiles_seq, dtype=torch.long)
print("Padded SMILES sequence shape:", padded_smiles_seq.shape)

def atom_features(atom):
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

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        bt.real, 
        bond.GetIsConjugated(),
        bond.IsInRing(), 
        bond.GetStereo().real, 
    ]

def graph_to_seq(smiles, atom_feat_fn=atom_features, bond_feat_fn=bond_features):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    atoms = mol.GetAtoms()
    atom_features_list = [atom_feat_fn(atom) for atom in atoms]
    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [(start, end), (end, start)]
        edge_features_list += [bond_feat_fn(bond), bond_feat_fn(bond)]
    node_features_tensor = torch.tensor(atom_features_list, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features_tensor = torch.tensor(edge_features_list, dtype=torch.float)
    return node_features_tensor, edge_index_tensor, edge_features_tensor

graph_node_features = []
graph_edge_indices = []
graph_edge_features = []

for smiles in smiles_list:
    node_feats, edge_idxs, edge_feats = graph_to_seq(smiles)
    if node_feats is not None:
        graph_node_features.append(node_feats)
        graph_edge_indices.append(edge_idxs)
        graph_edge_features.append(edge_feats)

max_atoms = max(len(node_feats) for node_feats in graph_node_features)

def pad_node_features(node_features, max_atoms, n_atom_features):
    padded_node_features = []
    for features_list in node_features:
        padding_length = max_atoms - len(features_list)
        if padding_length > 0:
            padding = torch.zeros(padding_length, n_atom_features)
            padded_features_list = torch.cat((features_list, padding), dim=0)
        else:
            padded_features_list = features_list
        padded_node_features.append(padded_features_list)
    return torch.stack(padded_node_features)

n_atom_features = len(graph_node_features[0][0])  
padded_node_features = pad_node_features(graph_node_features, max_atoms, n_atom_features)
reshaped_node_features = torch.flatten(padded_node_features, start_dim=1)
print("Reshaped node features shape:", reshaped_node_features.shape)

weight_padded_smiles_seq = 1
weight_reshaped_node_features = 1
weight_condition_features = 1

weighted_padded_smiles_seq = padded_smiles_seq * weight_padded_smiles_seq
weighted_reshaped_node_features = reshaped_node_features * weight_reshaped_node_features
weighted_condition_features = condition_features * weight_condition_features

print("Padded SMILES sequence shape:", padded_smiles_seq.shape)
print("Reshaped node features shape:", reshaped_node_features.shape)
print("Condition features shape:", condition_features.shape)

feature_smiles = padded_smiles_seq
feature_moleculargraph = reshaped_node_features
feature_smiles_condition = torch.cat((padded_smiles_seq, condition_features), dim=1)
feature_moleculargraph_condition = torch.cat((reshaped_node_features, condition_features), dim=1)

np.save('input_features/feature_smiles.npy', feature_smiles)
np.save('input_features/feature_moleculargraph.npy', feature_moleculargraph)
np.save('input_features/feature_smiles_condition.npy', feature_smiles_condition)
np.save('input_features/feature_moleculargraph_condition.npy', feature_moleculargraph_condition)

AFP_512 = pd.read_csv('AFP_512.csv', encoding='utf-8')
ECFP4_512 = pd.read_csv('ECFP4_512.csv', encoding='utf-8')
ECFP6_512 = pd.read_csv('ECFP6_512.csv', encoding='utf-8')
MACCS = pd.read_csv('MACCS.csv', encoding='utf-8')

AFP_512 = AFP_512.apply(pd.to_numeric, errors='coerce').fillna(0)
ECFP4_512 = ECFP4_512.apply(pd.to_numeric, errors='coerce').fillna(0)
ECFP6_512 = ECFP6_512.apply(pd.to_numeric, errors='coerce').fillna(0)
MACCS = MACCS.apply(pd.to_numeric, errors='coerce').fillna(0)

# The slicing [:condition_features.shape[0]] will now correctly use the filtered size (e.g., 1272)
AFP_512_features = torch.tensor(AFP_512.values, dtype=torch.float)[valid_indices]
ECFP4_512_features = torch.tensor(ECFP4_512.values, dtype=torch.float)[valid_indices]
ECFP6_512_features = torch.tensor(ECFP6_512.values, dtype=torch.float)[valid_indices]
MACCS_features = torch.tensor(MACCS.values, dtype=torch.float)[valid_indices]

X_combined_F_AFP_512 = AFP_512_features
X_combined_F_ECFP4_512 = ECFP4_512_features
X_combined_F_ECFP6_512 = ECFP6_512_features
X_combined_F_MACCS = MACCS_features

X_combined_F_AFP_512_C = torch.cat((condition_features, AFP_512_features), dim=1)
X_combined_F_ECFP4_512_C = torch.cat((condition_features, ECFP4_512_features), dim=1)
X_combined_F_ECFP6_512_C = torch.cat((condition_features, ECFP6_512_features), dim=1)
X_combined_F_MACCS_C = torch.cat((condition_features, MACCS_features), dim=1)

np.save('input_features/AFP_512.npy', X_combined_F_AFP_512)
np.save('input_features/ECFP4_512.npy', X_combined_F_ECFP4_512)
np.save('input_features/ECFP6_512.npy', X_combined_F_ECFP6_512)
np.save('input_features/MACCS.npy', X_combined_F_MACCS)
np.save('input_features/AFP_512_condition.npy', X_combined_F_AFP_512_C)
np.save('input_features/ECFP4_512_condition.npy', X_combined_F_ECFP4_512_C)
np.save('input_features/ECFP6_512_condition.npy', X_combined_F_ECFP6_512_C)
np.save('input_features/MACCS_condition.npy', X_combined_F_MACCS_C)

PubChemFP = pd.read_csv('PubChemFP.csv', encoding='utf-8')
PubChemFP = PubChemFP.apply(pd.to_numeric, errors='coerce').fillna(0)
# Also filter PubChem features using the same valid_indices
PubChemFP_features = torch.tensor(PubChemFP.values, dtype=torch.float)[valid_indices]
X_combined_F_PubChemFP = PubChemFP_features
X_combined_F_PubChemFP_C = torch.cat((condition_features, PubChemFP_features), dim=1)
np.save('input_features/PubChemFP.npy', X_combined_F_PubChemFP)
np.save('input_features/PubChemFP_condition.npy', X_combined_F_PubChemFP_C)
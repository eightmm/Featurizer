# Feature Types Documentation

## Molecular Features

### Fingerprints

#### Morgan Fingerprint
- **Size**: 2048 bits
- **Radius**: 2 (ECFP4-like)
- **Type**: Circular fingerprint
- **Use case**: Similarity searching, QSAR modeling

#### MACCS Keys
- **Size**: 167 bits
- **Type**: Structural keys
- **Use case**: Substructure searching, clustering

#### RDKit Fingerprint
- **Size**: 2048 bits
- **Type**: Daylight-like topological fingerprint
- **Use case**: General purpose molecular comparison

#### Atom Pair Fingerprint
- **Size**: 2048 bits
- **Type**: Atom pair descriptors
- **Use case**: Capturing atomic environments

#### Topological Torsion Fingerprint
- **Size**: 2048 bits
- **Type**: Four-atom torsion patterns
- **Use case**: Conformational analysis

#### 2D Pharmacophore Fingerprint
- **Size**: 1024 bits
- **Type**: Pharmacophoric features
- **Use case**: Pharmacophore-based screening

### Graph Representations

#### Node Features (per atom)
- **Dimensions**: Variable based on molecule size
- **Features**:
  - Atom type (one-hot)
  - Degree
  - Formal charge
  - Hybridization
  - Aromaticity
  - Ring membership

#### Edge Features (per bond)
- **Dimensions**: Variable based on molecule size
- **Features**:
  - Bond type (one-hot)
  - Conjugation
  - Ring membership
  - Stereochemistry

## Protein Features

### Sequence-based Features
- **AAC**: Amino Acid Composition (20 features)
- **DPC**: Dipeptide Composition (400 features)
- **CTD**: Composition, Transition, Distribution (147 features)
- **QSO**: Quasi-Sequence-Order (100 features)
- **PAAC**: Pseudo Amino Acid Composition (50 features)

### Structure-based Features
- **Secondary Structure**: Alpha helix, beta sheet, coil content
- **Solvent Accessibility**: Buried/exposed residue ratios
- **Contact Map**: Residue-residue contact patterns

### Functional Features
- **PSSM**: Position-Specific Scoring Matrix
- **Hydrophobicity**: Hydropathy profiles
- **Charge Distribution**: Positive/negative charge patterns

## Usage Examples

### Extracting Specific Features

```python
from featurizer import MoleculeFeaturizer, ProteinFeaturizer

# Molecule features
mol_featurizer = MoleculeFeaturizer()
mol_features = mol_featurizer.get_feature("CCO")

# Access individual components
descriptors = mol_features['descriptor']  # Shape: [40]
morgan_fp = mol_features['morgan']       # Shape: [2048]
maccs_keys = mol_features['maccs']       # Shape: [167]

# Protein features
prot_featurizer = ProteinFeaturizer()
prot_features = prot_featurizer.get_feature("MKFLILLFNILCLFPVLAAD")

# Access sequence features
aac = prot_features['aac']               # Shape: [20]
dpc = prot_features['dpc']               # Shape: [400]
```

### Graph Features for GNN

```python
# Get graph representation
graph = mol_featurizer.get_graph("c1ccccc1")

node_features = graph['node_feat']  # Shape: [n_atoms, n_features]
edge_features = graph['edge_feat']  # Shape: [n_edges, e_features]
edge_index = graph['edge_index']    # Shape: [2, n_edges]
```

## Feature Selection Guidelines

### For Machine Learning
- **Small datasets (<1000)**: Use descriptors + MACCS keys
- **Medium datasets (1000-10000)**: Add Morgan fingerprints
- **Large datasets (>10000)**: Use all fingerprints or graph features

### For Specific Applications
- **Similarity search**: Morgan or MACCS fingerprints
- **ADMET prediction**: Descriptors + selected fingerprints
- **Activity prediction**: Morgan + descriptors
- **Deep learning**: Graph representations

## Performance Considerations

### Memory Usage
- Descriptors: ~160 bytes per molecule
- Fingerprints: ~10 KB per molecule (all types)
- Graph features: Variable (depends on molecule size)

### Speed
- Descriptors: ~1ms per molecule
- Fingerprints: ~5ms per molecule (all types)
- Graph features: ~10ms per molecule

### Batch Processing
```python
# Efficient batch processing
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
batch_features = [mol_featurizer.get_feature(smi) for smi in smiles_list]

# Stack for neural network input
import torch
descriptors_batch = torch.stack([f['descriptor'] for f in batch_features])
```
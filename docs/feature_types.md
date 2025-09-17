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

### Sequence Features (`get_sequence_features()`)
- **Residue Types**: Integer encoding for 20 amino acids + UNK
- **One-hot Encoding**: 21-dimensional one-hot vectors per residue
- **Number of Residues**: Total residue count

### Geometric Features (`get_geometric_features()`)
- **Dihedral Angles**: Phi, psi, omega, and chi angles
- **Has Chi Angles**: Boolean flags for side chain torsions
- **Backbone Curvature**: Local backbone curvature measurements
- **Backbone Torsion**: Local backbone torsion angles
- **Self Distances**: Intra-residue atomic distances
- **Self Vectors**: Intra-residue directional vectors
- **Coordinates**: 3D coordinates for backbone and sidechain atoms

### SASA Features (`get_sasa_features()`)
- **Solvent Accessible Surface Area**: 10 components per residue
  - Total SASA
  - Polar/apolar SASA
  - Backbone/sidechain SASA
  - Relative SASA values

### Contact Features (`get_contact_map(cutoff)`)
- **Customizable Distance Threshold**: Any distance cutoff in Ångströms (default: 8.0)
- **Contact Matrix**: Binary contact map at specified cutoff
- **Distance Matrix**: CA-CA, SC-SC, CA-SC, SC-CA distances between residues
- **Edge Indices**: Residue pairs within distance cutoff
- **Common thresholds**:
  - 4.5 Å: Very close contacts (e.g., hydrogen bonds, salt bridges)
  - 8.0 Å: Standard contacts (typical protein interactions)
  - 12.0 Å: Extended interactions (long-range effects)

### Node Features (`get_node_features()`)
Combines all per-residue features:
- Residue type and one-hot encoding
- Terminal flags (N-terminal, C-terminal)
- Geometric features (dihedrals, curvature, torsion)
- SASA values
- Local coordinate frames

### Edge Features (`get_edge_features()`)
- **Spatial Distances**: CA-CA, SC-SC, CA-SC, SC-CA distances
- **Relative Position Encoding**: Sequence separation and spatial arrangement
- **Orientation Vectors**: 3D unit vectors between residues
- **Contact Indicators**: Binary contact flags

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
prot_featurizer = ProteinFeaturizer("protein.pdb")

# Get specific feature types
sequence = prot_featurizer.get_sequence_features()
geometry = prot_featurizer.get_geometric_features()
sasa = prot_featurizer.get_sasa_features()
contacts = prot_featurizer.get_contact_map(cutoff=8.0)
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
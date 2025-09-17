# Protein Features Documentation

## Overview
The protein featurizer extracts structural and sequence features from PDB files, providing residue-level and interaction features for machine learning applications on protein structures.

## Feature Extraction Methods

### 1. Sequence Features (`get_sequence_features()`)

Returns basic sequence information and encoding:

```python
{
    'residue_types': torch.Tensor,     # Shape: [n_residues] - Integer encoding (0-20)
    'residue_one_hot': torch.Tensor,   # Shape: [n_residues, 21] - One-hot vectors
    'num_residues': int                 # Total number of residues
}
```

**Residue encoding:**
- 0-19: Standard 20 amino acids
- 20: Unknown/non-standard residue

### 2. Geometric Features (`get_geometric_features()`)

Extracts 3D structural features:

```python
{
    'dihedrals': torch.Tensor,         # Shape: [n_residues, 4] - Phi, psi, omega, chi1
    'has_chi_angles': torch.Tensor,    # Shape: [n_residues, 4] - Boolean flags
    'backbone_curvature': torch.Tensor,# Shape: [n_residues] - Local curvature
    'backbone_torsion': torch.Tensor,  # Shape: [n_residues] - Local torsion
    'self_distances': torch.Tensor,    # Shape: [n_residues, 10] - Intra-residue distances
    'self_vectors': torch.Tensor,      # Shape: [n_residues, 30] - Direction vectors
    'coordinates': torch.Tensor        # Shape: [n_residues, 15, 3] - 3D coordinates
}
```

**Coordinate indices:**
- 0-3: Backbone atoms (N, CA, C, O)
- 4-14: Sidechain atoms (CB, CG, CD, CE, CZ, etc.)
- 14: Sidechain centroid

### 3. SASA Features (`get_sasa_features()`)

Solvent Accessible Surface Area analysis:

```python
torch.Tensor  # Shape: [n_residues, 10]
```

**10 SASA components per residue:**
1. Total SASA
2. Polar SASA
3. Apolar SASA
4. Backbone SASA
5. Sidechain SASA
6. Relative total SASA
7. Relative polar SASA
8. Relative apolar SASA
9. Relative backbone SASA
10. Relative sidechain SASA

### 4. Contact Map (`get_contact_map(cutoff)`)

Residue-residue interaction analysis with customizable distance threshold:

```python
{
    'adjacency_matrix': torch.Tensor,  # Shape: [n_res, n_res] - Binary contacts
    'distance_matrix': torch.Tensor,   # Shape: [n_res, n_res] - Distances (Å)
    'edges': tuple,                    # (src_indices, dst_indices)
    'distances': torch.Tensor,         # Shape: [n_edges] - Edge distances
    'cutoff': float                    # Distance threshold used
}
```

**Distance calculations include:**
- CA-CA: Alpha carbon distances
- SC-SC: Sidechain centroid distances
- CA-SC: Mixed distances
- SC-CA: Reverse mixed distances

**Common distance thresholds:**
- **4.5 Å**: Close contacts (H-bonds, salt bridges)
- **8.0 Å**: Standard protein interactions (default)
- **12.0 Å**: Extended/long-range interactions

### 5. Relative Position Encoding (`get_relative_position()`)

Encodes spatial relationships between residue pairs:

```python
torch.Tensor  # Shape: [n_residues, n_residues, 32]
```

Uses sinusoidal encoding of sequence separation and spatial distance.

### 6. Terminal Flags (`get_terminal_flags()`)

Identifies chain termini:

```python
torch.Tensor  # Shape: [n_residues, 2]
```
- Column 0: N-terminal flag
- Column 1: C-terminal flag

### 7. Node Features (`get_node_features()`)

Comprehensive per-residue features combining all individual features:

```python
{
    'scalar': torch.Tensor,  # All scalar features concatenated
    'vector': torch.Tensor,  # All vector features
    'sasa': torch.Tensor,    # SASA features
    'dihedrals': torch.Tensor,
    'terminal_flags': torch.Tensor
}
```

### 8. Edge Features (`get_edge_features(distance_cutoff)`)

Inter-residue interaction features for edges within cutoff:

```python
{
    'edges': tuple,                    # (src, dst) indices
    'distance': torch.Tensor,          # Shape: [n_edges] - Distances
    'relative_position': torch.Tensor, # Shape: [n_edges, 32] - Position encoding
    'vectors': torch.Tensor            # Shape: [n_edges, 3] - Unit vectors
}
```

### 9. All Features (`get_all_features()`)

Returns complete feature set in one call:

```python
{
    'node': dict,  # All node features
    'edge': dict,  # All edge features
    'metadata': {
        'num_residues': int,
        'distance_cutoff': float
    }
}
```

## Usage Examples

### Basic Feature Extraction
```python
from featurizer import ProteinFeaturizer

# Initialize with PDB file
featurizer = ProteinFeaturizer("protein.pdb")

# Extract individual feature types
sequence = featurizer.get_sequence_features()
geometry = featurizer.get_geometric_features()
sasa = featurizer.get_sasa_features()
```

### Working with Contact Maps
```python
# Different analysis scenarios
close_contacts = featurizer.get_contact_map(cutoff=4.5)
standard_contacts = featurizer.get_contact_map(cutoff=8.0)
extended_contacts = featurizer.get_contact_map(cutoff=12.0)

# Analyze contact patterns
adjacency = close_contacts['adjacency_matrix']
num_contacts = adjacency.sum(dim=1)  # Contacts per residue
```

### Graph Representation for GNNs
```python
# Get graph features
all_features = featurizer.get_all_features()
node_features = all_features['node']
edge_features = all_features['edge']

# Create graph with DGL
import dgl

src, dst = edge_features['edges']
g = dgl.graph((src, dst))
g.ndata['feat'] = node_features['scalar']
g.edata['feat'] = edge_features['vectors']
g.edata['dist'] = edge_features['distance']
```

### PDB Standardization
```python
# With automatic PDB cleaning (default)
featurizer = ProteinFeaturizer("messy.pdb", standardize=True)

# Skip standardization for pre-cleaned PDBs
featurizer = ProteinFeaturizer("clean.pdb", standardize=False)
```

### Batch Processing Multiple Proteins
```python
import glob

pdb_files = glob.glob("pdbs/*.pdb")
all_protein_features = []

for pdb_file in pdb_files:
    featurizer = ProteinFeaturizer(pdb_file)
    features = featurizer.get_all_features()
    all_protein_features.append(features)
```

### Efficient Feature Extraction
```python
# Instance mode - parse PDB once, extract multiple features
featurizer = ProteinFeaturizer("protein.pdb")

# All subsequent calls use cached structure
seq = featurizer.get_sequence_features()
geo = featurizer.get_geometric_features()
sasa = featurizer.get_sasa_features()
contacts = featurizer.get_contact_map(8.0)

# This is more efficient than creating new instances
```

## Feature Selection Guidelines

### By Application

#### Protein-Protein Interaction
- Contact maps with 8-12 Å cutoff
- SASA features (exposed residues)
- Edge features for interface

#### Protein Folding/Stability
- Geometric features (dihedrals, curvature)
- Contact maps with 4.5-8 Å cutoff
- SASA features

#### Function Prediction
- Sequence features
- Geometric features
- SASA (active site accessibility)

#### Structure Quality Assessment
- Geometric features (check unusual angles)
- Contact maps (check packing)
- Terminal flags (check completeness)

### By Model Type

#### Graph Neural Networks
- Node features (per-residue)
- Edge features (interactions)
- Contact maps as adjacency

#### Traditional ML (RF, SVM)
- Flatten geometric features
- SASA statistics
- Contact density metrics

#### Sequence-based Models
- Sequence features only
- Add position encoding if needed

## Performance Considerations

### Speed
- **PDB parsing**: 10-100ms (depends on size)
- **SASA calculation**: 50-200ms
- **Contact map**: 5-20ms
- **All features**: 100-500ms total

### Memory Usage
- **Small protein (<100 residues)**: ~1 MB
- **Medium protein (100-500 residues)**: ~5-10 MB
- **Large protein (>500 residues)**: ~20-50 MB

### Optimization Tips

1. **Use instance mode**: Parse PDB once, extract multiple features
2. **Cache results**: Store computed features for reuse
3. **Selective extraction**: Only compute needed features
4. **Batch processing**: Process multiple proteins in parallel
5. **Pre-standardization**: Clean PDBs beforehand if processing many times

## Notes on PDB Standardization

The featurizer includes automatic PDB standardization that:
- Selects best model from multi-model structures
- Removes heteroatoms (water, ligands)
- Fixes common PDB format issues
- Ensures consistent chain naming

Disable with `standardize=False` if your PDBs are pre-processed.
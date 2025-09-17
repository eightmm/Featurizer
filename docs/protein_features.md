# Protein Features Documentation

## Overview
The protein featurizer extracts structural and sequence features from PDB files, providing residue-level and interaction features for machine learning applications on protein structures.

## Feature Extraction Methods

### Atom-Level Features

#### `get_atom_features()`

Extract atom-level tokenized features:

```python
token, coord = featurizer.get_atom_features("protein.pdb")
```

Returns:
- `token`: Atom type tokens (175 unique types for residue-atom combinations)
- `coord`: 3D coordinates for each atom

#### `get_atom_features_with_sasa()`

Get comprehensive atom features including SASA:

```python
features = featurizer.get_atom_features_with_sasa("protein.pdb")
```

Returns dictionary with:
- `token`: Atom type tokens
- `coord`: 3D coordinates
- `sasa`: Solvent accessible surface area per atom
- `residue_token`: Residue type for each atom
- `atom_element`: Element type
- `radius`: Atomic radii
- `metadata`: Additional atom information

### Residue-Level Features

### 1. Sequence Features (`get_sequence_features()`)

Returns basic sequence information and encoding:

```python
{
    'residue_types': torch.Tensor,     # Integer encoding (0-20)
    'residue_one_hot': torch.Tensor,   # One-hot vectors
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
    'dihedrals': torch.Tensor,         # Phi, psi, omega, chi angles
    'has_chi_angles': torch.Tensor,    # Boolean flags for chi angles
    'backbone_curvature': torch.Tensor,# Local curvature
    'backbone_torsion': torch.Tensor,  # Local torsion
    'self_distances': torch.Tensor,    # Intra-residue distances
    'self_vectors': torch.Tensor,      # Direction vectors
    'coordinates': torch.Tensor        # 3D coordinates
}
```

**Coordinate indices:**
- 0-3: Backbone atoms (N, CA, C, O)
- 4-14: Sidechain atoms (CB, CG, CD, CE, CZ, etc.)
- 14: Sidechain centroid

### 3. SASA Features (`get_sasa_features()`)

Solvent Accessible Surface Area analysis:

Returns a tensor containing SASA values for each residue.

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
    'adjacency_matrix': torch.Tensor,  # Binary contacts
    'distance_matrix': torch.Tensor,   # Distances in Ångströms
    'edges': tuple,                    # (src_indices, dst_indices)
    'distances': torch.Tensor,         # Edge distances
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

Returns a tensor encoding spatial relationships between residue pairs.

Uses sinusoidal encoding of sequence separation and spatial distance.

### 6. Terminal Flags (`get_terminal_flags()`)

Identifies chain termini:

Returns a tensor with N-terminal and C-terminal flags for each residue.

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
    'distance': torch.Tensor,          # Distances
    'relative_position': torch.Tensor, # Position encoding
    'vectors': torch.Tensor            # Unit vectors
}
```

### 9. Standard Features (`get_features()`)

Returns node and edge features in the standard format:

```python
node, edge = featurizer.get_features()

# node dictionary contains:
{
    'coord': torch.Tensor,                    # CA and SC coordinates
    'node_scalar_features': torch.Tensor,     # Scalar features per residue
    'node_vector_features': torch.Tensor      # Vector features per residue
}

# edge dictionary contains:
{
    'edges': tuple,                           # (src, dst) indices
    'edge_scalar_features': torch.Tensor,     # Scalar edge features
    'edge_vector_features': torch.Tensor      # Vector edge features
}
```

### 10. All Features (`get_all_features()`)

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

# Get standard node and edge features
node, edge = featurizer.get_features()

# Extract individual feature types
sequence = featurizer.get_sequence_features()
geometry = featurizer.get_geometric_features()
sasa = featurizer.get_sasa_features()
```

### Atom-Level Feature Extraction
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Get basic atom features
token, coord = featurizer.get_atom_features()
print(f"Number of atoms: {len(token)}")
print(f"Atom tokens shape: {token.shape}")
print(f"Coordinates shape: {coord.shape}")

# Get atom features with SASA
atom_features = featurizer.get_atom_features_with_sasa()
print(f"SASA per atom: {atom_features['sasa']}")
print(f"Atom radii: {atom_features['radius']}")

# Direct usage without class
from featurizer.protein_featurizer import get_protein_atom_features
token, coord = get_protein_atom_features("protein.pdb")
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
# Get standard features format
node, edge = featurizer.get_features()

# Access coordinates and features
coords = node['coord']  # CA and SC positions
node_scalar = node['node_scalar_features']
node_vector = node['node_vector_features']

edges = edge['edges']  # (src, dst) tuples
edge_scalar = edge['edge_scalar_features']
edge_vector = edge['edge_vector_features']

# Create graph with DGL
import dgl

src, dst = edges
g = dgl.graph((src, dst))
g.ndata['coord'] = coords
g.ndata['scalar_feat'] = node_scalar
g.ndata['vector_feat'] = node_vector
g.edata['scalar_feat'] = edge_scalar
g.edata['vector_feat'] = edge_vector
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
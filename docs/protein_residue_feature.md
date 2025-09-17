# Protein Residue-Level Features Documentation

## Overview
Comprehensive residue-level feature extraction for protein structure analysis and machine learning applications.

## Feature Extraction Methods

### Method Naming Convention
All residue-level methods have clear aliases for better clarity:
- Original: `get_sequence_features()` → Aliases: `get_residue_sequence()`, `get_residue_types()`
- Original: `get_geometric_features()` → Aliases: `get_residue_geometry()`, `get_residue_dihedrals()`
- Original: `get_sasa_features()` → Aliases: `get_residue_sasa()`, `get_residue_level_sasa()`
- Original: `get_contact_map()` → Aliases: `get_residue_contacts()`, `get_residue_contact_map()`
- Original: `get_features()` → Aliases: `get_residue_features()`, `get_residue_level_features()`

### 1. Standard Features (`get_features()` / `get_residue_features()`)

Returns node and edge features in standard format for graph neural networks.

```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")
node, edge = featurizer.get_features()

# Node features
coords = node['coord']                        # [n_residues, 2, 3] CA and SC coordinates
node_scalar = node['node_scalar_features']    # Scalar features per residue
node_vector = node['node_vector_features']    # Vector features per residue

# Edge features
edges = edge['edges']                         # (src, dst) indices
edge_scalar = edge['edge_scalar_features']    # Scalar edge features
edge_vector = edge['edge_vector_features']    # Vector edge features
```

### 2. Sequence Features (`get_sequence_features()`)

Basic sequence information and encoding.

```python
seq_features = featurizer.get_sequence_features()

# Returns:
{
    'residue_types': torch.Tensor,     # [n_residues] Integer encoding (0-20)
    'residue_one_hot': torch.Tensor,   # [n_residues, 21] One-hot vectors
    'num_residues': int                 # Total number of residues
}
```

**Residue Encoding:**
- 0-19: Standard 20 amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
- 20: Unknown/non-standard residue

### 3. Geometric Features (`get_geometric_features()`)

3D structural features and measurements.

```python
geo_features = featurizer.get_geometric_features()

# Returns:
{
    'dihedrals': torch.Tensor,         # Backbone and sidechain angles
    'has_chi_angles': torch.Tensor,    # Boolean flags for chi angles
    'backbone_curvature': torch.Tensor,# Local curvature
    'backbone_torsion': torch.Tensor,  # Local torsion
    'self_distances': torch.Tensor,    # Intra-residue distances
    'self_vectors': torch.Tensor,      # Direction vectors
    'coordinates': torch.Tensor        # [n_residues, 15, 3] All atom coords
}
```

**Coordinate Indices:**
- 0-3: Backbone atoms (N, CA, C, O)
- 4-13: Sidechain atoms (CB, CG, CD, CE, CZ, etc.)
- 14: Sidechain centroid

**Dihedral Angles:**
- Phi (φ): C(-1) - N - CA - C
- Psi (ψ): N - CA - C - N(+1)
- Omega (ω): CA - C - N(+1) - CA(+1)
- Chi angles (χ1-χ4): Sidechain rotamers

### 4. SASA Features (`get_sasa_features()`)

Solvent Accessible Surface Area analysis.

```python
sasa = featurizer.get_sasa_features()  # [n_residues, 10]
```

**10 SASA Components per Residue:**
1. Total SASA
2. Polar SASA
3. Apolar SASA
4. Backbone SASA
5. Sidechain SASA
6. Relative total SASA (normalized by max possible)
7. Relative polar SASA
8. Relative apolar SASA
9. Relative backbone SASA
10. Relative sidechain SASA

### 5. Contact Map (`get_contact_map()`)

Residue-residue interaction analysis with customizable distance threshold.

```python
contacts = featurizer.get_contact_map(cutoff=8.0)

# Returns:
{
    'adjacency_matrix': torch.Tensor,  # [n_res, n_res] Binary contacts
    'distance_matrix': torch.Tensor,   # [n_res, n_res] Distances in Å
    'edges': tuple,                    # (src_indices, dst_indices)
    'edge_distances': torch.Tensor,    # Edge distances
    'interaction_vectors': torch.Tensor # Direction vectors between residues
}
```

**Distance Calculations:**
- CA-CA: Alpha carbon distances
- SC-SC: Sidechain centroid distances
- CA-SC: Mixed distances
- SC-CA: Reverse mixed distances

**Common Thresholds:**
```python
# Different analysis scenarios
close_contacts = featurizer.get_contact_map(cutoff=4.5)   # H-bonds, salt bridges
standard_contacts = featurizer.get_contact_map(cutoff=8.0) # Standard interactions
extended_contacts = featurizer.get_contact_map(cutoff=12.0) # Long-range interactions
```

### 6. Node Features (`get_node_features()`)

Comprehensive per-residue features.

```python
node_features = featurizer.get_node_features()

# Returns:
{
    'coordinates': torch.Tensor,       # [n_residues, 2, 3] CA and SC coords
    'scalar_features': torch.Tensor,   # All scalar features concatenated
    'vector_features': torch.Tensor    # All vector features
}
```

### 7. Edge Features (`get_edge_features()`)

Inter-residue interaction features.

```python
edge_features = featurizer.get_edge_features(distance_cutoff=8.0)

# Returns:
{
    'edges': tuple,                    # (src, dst) indices
    'scalar_features': torch.Tensor,   # Distance-based features
    'vector_features': torch.Tensor    # Direction vectors
}
```

### 8. Relative Position Encoding (`get_relative_position()`)

Encodes spatial relationships between residue pairs.

```python
rel_pos = featurizer.get_relative_position(cutoff=32)
# Returns one-hot encoded relative positions
```

### 9. Terminal Flags (`get_terminal_flags()`)

Identifies chain termini.

```python
terminals = featurizer.get_terminal_flags()

# Returns:
{
    'n_terminal': torch.Tensor,  # Binary flags for N-terminus
    'c_terminal': torch.Tensor   # Binary flags for C-terminus
}
```

### 10. All Features (`get_all_features()`)

Complete feature set in one call.

```python
all_features = featurizer.get_all_features()

# Returns:
{
    'node': dict,  # All node features
    'edge': dict,  # All edge features
    'metadata': {
        'num_residues': int,
        'distance_cutoff': float
    }
}

# Optional: Save to file
all_features = featurizer.get_all_features(save_to='features.pt')
```

## Usage Examples

### Basic Feature Extraction
```python
from featurizer import ProteinFeaturizer

# Initialize with PDB file
featurizer = ProteinFeaturizer("protein.pdb")

# Get standard format (multiple aliases available)
node, edge = featurizer.get_features()  # Original
node, edge = featurizer.get_residue_features()  # Clearer
node, edge = featurizer.get_residue_level_features()  # Most explicit

# Extract specific features (with clearer aliases)
sequence = featurizer.get_residue_sequence()  # or get_sequence_features()
geometry = featurizer.get_residue_geometry()  # or get_geometric_features()
sasa = featurizer.get_residue_sasa()  # or get_sasa_features()
```

### Graph Neural Network Integration

#### DGL Integration
```python
import dgl

node, edge = featurizer.get_features()

# Create graph
src, dst = edge['edges']
g = dgl.graph((src, dst))

# Add features
g.ndata['coord'] = node['coord']
g.ndata['scalar_feat'] = node['node_scalar_features']
g.ndata['vector_feat'] = node['node_vector_features']
g.edata['scalar_feat'] = edge['edge_scalar_features']
g.edata['vector_feat'] = edge['edge_vector_features']
```

#### PyTorch Geometric Integration
```python
from torch_geometric.data import Data

node, edge = featurizer.get_features()

data = Data(
    x=node['node_scalar_features'],
    edge_index=edge['edges'],
    edge_attr=edge['edge_scalar_features'],
    pos=node['coord'].reshape(-1, 6)  # Flatten CA and SC coords
)
```

### Contact Analysis
```python
# Analyze contact patterns at different thresholds
featurizer = ProteinFeaturizer("protein.pdb")

for cutoff in [4.5, 8.0, 12.0]:
    contacts = featurizer.get_residue_contacts(cutoff=cutoff)  # or get_contact_map()
    adjacency = contacts['adjacency_matrix']
    num_contacts = adjacency.sum(dim=1)

    print(f"Cutoff {cutoff}Å: avg contacts = {num_contacts.mean():.1f}")
```

### Batch Processing
```python
import glob

pdb_files = glob.glob("pdbs/*.pdb")
all_features = []

for pdb_file in pdb_files:
    featurizer = ProteinFeaturizer(pdb_file)
    features = featurizer.get_all_features()
    all_features.append(features)
```

## Efficient Feature Extraction

The ProteinFeaturizer class parses PDB once and caches results:

```python
# Parse PDB once
featurizer = ProteinFeaturizer("protein.pdb")

# All subsequent calls use cached structure
seq = featurizer.get_sequence_features()  # No re-parsing
geo = featurizer.get_geometric_features()  # Uses cache
sasa = featurizer.get_sasa_features()      # Uses cache
contacts = featurizer.get_contact_map(8.0) # Uses cache
```

## PDB Standardization

Automatic PDB standardization (can be disabled):

```python
# With automatic standardization (default)
featurizer = ProteinFeaturizer("messy.pdb", standardize=True)

# Skip standardization for pre-cleaned PDBs
featurizer = ProteinFeaturizer("clean.pdb", standardize=False)

# Keep hydrogens during standardization
featurizer = ProteinFeaturizer("protein.pdb", keep_hydrogens=True)
```

Standardization includes:
- Selecting best model from multi-model structures
- Removing heteroatoms (water, ligands)
- Fixing common PDB format issues
- Ensuring consistent chain naming

## Feature Selection Guidelines

### By Application

#### Protein Folding/Stability
- Geometric features (dihedrals, curvature)
- Contact maps with 4.5-8 Å cutoff
- SASA features

#### Protein-Protein Interaction
- Contact maps with 8-12 Å cutoff
- SASA features (exposed residues)
- Edge features for interface

#### Function Prediction
- Sequence features
- Geometric features
- SASA (active site accessibility)

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

## Performance

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
1. Use instance mode: Parse PDB once, extract multiple features
2. Cache results: Store computed features for reuse
3. Selective extraction: Only compute needed features
4. Batch processing: Process multiple proteins in parallel
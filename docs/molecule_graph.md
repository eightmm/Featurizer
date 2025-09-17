# Molecule Graph Representation Documentation

## Overview
Graph representations of molecules for Graph Neural Networks (GNNs) with comprehensive atom and bond features.

## Graph Features (`get_graph()`)

Converts molecular structures into graph format with node (atom) and edge (bond) features.

### Usage
```python
from featurizer import MoleculeFeaturizer

# Initialize with molecule
featurizer = MoleculeFeaturizer("CCO")
node, edge = featurizer.get_graph()

# Access features
node_features = node['node_feats']  # [n_atoms, 122]
edge_features = edge['edge_feats']  # [n_edges, 44]
coordinates = node['coords']        # [n_atoms, 3]
edge_indices = edge['edges']        # [2, n_edges]
```

## Node Features (122 dimensions)

### Atom Type (44 dimensions)
One-hot encoding for common atom types:
- C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, As, Al, I, B, V, K, Tl, Yb, Sb, Sn, Ag, Pd, Co, Se, Ti, Zn, H, Li, Ge, Cu, Au, Ni, Cd, In, Mn, Zr, Cr, Pt, Hg, Pb, Unknown

### Atom Degree (11 dimensions)
One-hot encoding for number of bonds:
- Degrees 0-10

### Formal Charge (11 dimensions)
One-hot encoding for formal charge:
- Charges from -5 to +5

### Chiral Tag (5 dimensions)
Stereochemistry information:
- Unspecified, CW, CCW, Other, None

### Hybridization (9 dimensions)
Orbital hybridization state:
- s, sp, sp², sp³, sp³d, sp³d², unspecified, other, none

### Aromaticity (2 dimensions)
- Is aromatic (binary)
- Is not aromatic (binary)

### Hydrogen Bonds (6 dimensions)
- Total H count (integer, 0-5)
- Number of implicit H (integer, 0-5)

### Radical Electrons (5 dimensions)
One-hot encoding for radical electrons (0-4)

### Ring Membership (10 dimensions)
Binary flags for membership in rings of size 3-12

### Chemical Properties (9 dimensions)
Binary flags:
- Is donor
- Is acceptor
- Is aromatic
- Is heteroatom
- sp hybridization
- sp² hybridization
- sp³ hybridization
- Is ring member
- Is in ring of size 3, 4, 5, 6

## Edge Features (44 dimensions)

### Bond Type (5 dimensions)
One-hot encoding:
- Single, Double, Triple, Aromatic, None

### Conjugated (2 dimensions)
Binary flag for conjugation

### Ring Membership (2 dimensions)
Binary flag for ring membership

### Stereochemistry (7 dimensions)
Bond stereochemistry:
- None, Any, E, Z, Cis, Trans, Other

### Bond Distance (1 dimension)
3D distance between atoms (when conformer available)

### Additional Features (27 dimensions)
Extended bond descriptors for richer representations

## 3D Coordinates

When available, preserves 3D molecular geometry:

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Generate 3D conformer
mol = Chem.MolFromSmiles("CCO")
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

# Extract with 3D coordinates
featurizer = MoleculeFeaturizer(mol)
node, edge = featurizer.get_graph()
coords = node['coords']  # 3D coordinates [n_atoms, 3]
```

## Graph Construction

### Edge List Format
```python
edge['edges']  # [2, n_edges]
# Format: [[source_nodes], [target_nodes]]
# Undirected: both (i→j) and (j→i) are included
```

### Integration with DGL
```python
import dgl
import torch

featurizer = MoleculeFeaturizer("c1ccccc1")
node, edge = featurizer.get_graph()

# Create DGL graph
src, dst = edge['edges']
g = dgl.graph((src, dst))

# Add features
g.ndata['feat'] = node['node_feats']
g.edata['feat'] = edge['edge_feats']
g.ndata['coords'] = node['coords']

# Use in GNN
print(f"Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
```

### Integration with PyTorch Geometric
```python
from torch_geometric.data import Data

featurizer = MoleculeFeaturizer("CCO")
node, edge = featurizer.get_graph()

data = Data(
    x=node['node_feats'],
    edge_index=edge['edges'],
    edge_attr=edge['edge_feats'],
    pos=node['coords']
)
```

## Batch Processing

```python
from featurizer import MoleculeFeaturizer
import torch

featurizer = MoleculeFeaturizer()
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

# Process batch
graphs = []
for smi in smiles_list:
    featurizer = MoleculeFeaturizer(smi)
    node, edge = featurizer.get_graph()
    graphs.append((node, edge))

# Collate for batching (example with DGL)
import dgl

dgl_graphs = []
for node, edge in graphs:
    g = dgl.graph((edge['edges'][0], edge['edges'][1]))
    g.ndata['feat'] = node['node_feats']
    g.edata['feat'] = edge['edge_feats']
    dgl_graphs.append(g)

batched_graph = dgl.batch(dgl_graphs)
```

## Feature Normalization

All continuous features are normalized:
- Charges: shifted and scaled to [0, 1]
- Distances: normalized by typical bond lengths
- Counts: divided by reasonable maximum values

## Applications

### Drug Discovery
- Molecular property prediction
- Drug-target interaction
- ADMET prediction

### Materials Science
- Material property prediction
- Molecular design
- Reaction prediction

### Chemical Informatics
- Similarity searching
- Scaffold hopping
- Molecular clustering

## Performance
- **Small molecules (<50 atoms)**: ~5ms
- **Medium molecules (50-100 atoms)**: ~10ms
- **Large molecules (>100 atoms)**: ~20ms

## Advanced Usage

### Custom Featurization
```python
# Initialize with molecule
featurizer = MoleculeFeaturizer(mol)

# Get both features and graph
features = featurizer.get_feature()  # Descriptors + fingerprints
node, edge = featurizer.get_graph()  # Graph representation

# Combine for multi-modal learning
combined = {
    'descriptors': features['descriptor'],
    'fingerprints': features['morgan'],
    'graph': (node, edge)
}
```

### Substructure Graphs
```python
from rdkit import Chem

# Extract substructure
mol = Chem.MolFromSmiles("c1ccccc1CCO")
pattern = Chem.MolFromSmarts("c1ccccc1")
matches = mol.GetSubstructMatches(pattern)

# Get full graph
featurizer = MoleculeFeaturizer(mol)
node, edge = featurizer.get_graph()

# Filter for substructure
subgraph_atoms = matches[0]
subgraph_features = node['node_feats'][list(subgraph_atoms)]
```
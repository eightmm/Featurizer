# Molecule Features Documentation

## Overview
The molecule featurizer provides comprehensive feature extraction for small molecules, including descriptors, fingerprints, and graph representations for machine learning applications.

## Feature Types

### 1. Molecular Descriptors (`get_feature()['descriptor']`)
**40 normalized molecular properties** - See [Molecular Descriptors Reference](molecular_descriptors.md) for complete list

#### Categories:
- **Physicochemical** (12): MW, LogP, TPSA, rotatable bonds, HBD/HBA, etc.
- **Topological** (9): BalabanJ, BertzCT, Chi indices, Kappa shapes
- **Electronic** (4): MolMR, LabuteASA, radical/valence electrons
- **Structural** (8): Ring systems, saturated/aromatic rings
- **Drug-likeness** (5): Lipinski, QED, heavy atoms, Fsp3
- **Other** (2): Formal charge, Chi0n

### 2. Molecular Fingerprints

#### Morgan Fingerprint (`get_feature()['morgan']`)
- **Size**: 2048 bits
- **Radius**: 2 (ECFP4-like)
- **Type**: Circular fingerprint
- **Use case**: Similarity searching, QSAR modeling

#### Morgan Count Fingerprint (`get_feature()['morgan_count']`)
- **Size**: 2048 dimensions
- **Type**: Count-based Morgan fingerprint
- **Use case**: Preserves frequency information

#### Feature Morgan (`get_feature()['feature_morgan']`)
- **Size**: 2048 bits
- **Type**: Feature-based circular fingerprint
- **Use case**: Pharmacophore-aware similarity

#### MACCS Keys (`get_feature()['maccs']`)
- **Size**: 167 bits
- **Type**: Structural keys
- **Use case**: Substructure searching, clustering

#### RDKit Fingerprint (`get_feature()['rdkit']`)
- **Size**: 2048 bits
- **Type**: Daylight-like topological fingerprint
- **Use case**: General purpose molecular comparison

#### Atom Pair Fingerprint (`get_feature()['atom_pair']`)
- **Size**: 2048 bits
- **Type**: Atom pair descriptors
- **Use case**: Capturing atomic environments

#### Topological Torsion (`get_feature()['topological_torsion']`)
- **Size**: 2048 bits
- **Type**: Four-atom torsion patterns
- **Use case**: Conformational analysis

#### 2D Pharmacophore (`get_feature()['pharmacophore2d']`)
- **Size**: 1024 bits
- **Type**: Pharmacophoric features
- **Use case**: Pharmacophore-based screening

### 3. Graph Representations (`get_graph()`)

Returns a tuple of `(node, edge)` dictionaries:

#### Node Features
```python
node = {
    'coords': torch.Tensor,      # Shape: [n_atoms, 3] - 3D coordinates
    'node_feats': torch.Tensor   # Shape: [n_atoms, 122] - Atom features
}
```

**122-dimensional atom features include:**
- **Atom type** (11): One-hot for H, C, N, O, S, P, F, Cl, Br, I, Unknown
- **Atomic properties** (8): Period, group, electronegativity, covalent radius
- **Connectivity** (6): Degree, implicit/explicit valence
- **Ring membership** (7): In ring, ring sizes (3-8)
- **Aromaticity** (1): Is aromatic
- **Hybridization** (5): sp, sp2, sp3, sp3d, sp3d2
- **Hydrogen bonding** (2): H-bond donor/acceptor
- **Formal charge** (1): Normalized charge
- **Stereochemistry** (2): R/S configuration
- **Extended features** (79): Additional chemical properties

#### Edge Features
```python
edge = {
    'edges': torch.Tensor,        # Shape: [2, n_edges] - Source/destination indices
    'edge_feats': torch.Tensor   # Shape: [n_edges, 44] - Bond features
}
```

**44-dimensional bond features include:**
- **Bond type** (4): Single, double, triple, aromatic
- **Stereochemistry** (3): E/Z, cis/trans
- **Conjugation** (1): Is conjugated
- **Ring membership** (1): In ring
- **Rotatability** (1): Is rotatable
- **Extended features** (34): Additional bond properties

## Usage Examples

### Basic Feature Extraction
```python
from featurizer import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()

# From SMILES
features = featurizer.get_feature("CCO")
descriptors = features['descriptor']  # Shape: [40]
morgan = features['morgan']          # Shape: [2048]

# From RDKit mol
from rdkit import Chem
mol = Chem.MolFromSmiles("c1ccccc1")
features = featurizer.get_feature(mol)
```

### Graph Features for GNN
```python
# Get graph representation
node, edge = featurizer.get_graph("CC(=O)O")

# Access features
atom_features = node['node_feats']  # [n_atoms, 122]
bond_features = edge['edge_feats']  # [n_edges, 44]
coordinates = node['coords']        # [n_atoms, 3]

# Create DGL graph
import dgl
src, dst = edge['edges'][0], edge['edges'][1]
g = dgl.graph((src, dst))
g.ndata['feat'] = atom_features
g.edata['feat'] = bond_features
```

### Working with 3D Structures
```python
from rdkit.Chem import AllChem

# Generate 3D conformer
mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

# Extract with 3D coordinates preserved
node, edge = featurizer.get_graph(mol)
coords_3d = node['coords']  # 3D coordinates maintained
```

### Batch Processing
```python
# Process multiple molecules from SDF
from rdkit import Chem

supplier = Chem.SDMolSupplier('molecules.sdf')
all_features = []

for mol in supplier:
    if mol is not None:
        features = featurizer.get_feature(mol)
        all_features.append(features)

# Stack descriptors for ML
import torch
descriptors = torch.stack([f['descriptor'] for f in all_features])
```

## Feature Selection Guidelines

### By Dataset Size
- **Small (<1000)**: Descriptors + MACCS keys
- **Medium (1000-10000)**: Add Morgan fingerprints
- **Large (>10000)**: All fingerprints or graph features

### By Application
- **Similarity search**: Morgan or MACCS fingerprints
- **ADMET prediction**: Descriptors + selected fingerprints
- **Activity prediction**: Morgan + descriptors
- **Deep learning**: Graph representations
- **Clustering**: MACCS or Morgan fingerprints

### By Model Type
- **Random Forest/XGBoost**: Descriptors + fingerprints
- **SVM**: Normalized descriptors + binary fingerprints
- **Neural Networks**: All features or graph representations
- **GNN**: Graph features only

## Performance Considerations

### Memory Usage
- **Descriptors**: ~160 bytes per molecule
- **Each fingerprint**: 256-2048 bytes
- **Graph features**: Variable (depends on molecule size)
- **Total (all features)**: ~10 KB per molecule

### Speed
- **Descriptors**: ~1ms per molecule
- **All fingerprints**: ~5ms per molecule
- **Graph features**: ~10ms per molecule
- **With 3D generation**: +50-200ms per molecule

### Optimization Tips
1. Cache featurizer instance for multiple molecules
2. Use batch processing for large datasets
3. Pre-generate 3D coordinates if needed repeatedly
4. Select only necessary feature types
5. Consider using sparse representations for fingerprints
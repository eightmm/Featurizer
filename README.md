# Featurizer

A comprehensive Python package for extracting features from both **molecules** (small molecules/drugs) and **proteins** for machine learning applications, with special support for graph neural networks.

## 🌟 Key Features

### 🧪 Molecule Featurizer
- **Dual Input Support**: RDKit mol objects (primary) or SMILES strings
- **3D Coordinate Preservation**: Maintains and generates 3D structures
- **Graph Representations**: Node and edge features for GNN models
- **Comprehensive Fingerprints**: Multiple molecular fingerprint types
- **Molecular Descriptors**: Physicochemical and drug-likeness properties

### 🧬 Protein Featurizer
- **PDB File Processing**: Automatic standardization and cleaning
- **Residue-Level Features**: Node features per amino acid residue
- **Interaction Features**: Edge features for residue-residue interactions
- **SASA Calculations**: Solvent accessible surface area analysis
- **3D Structure Features**: Geometric and structural properties

## 📦 Installation

### Install from GitHub

```bash
pip install git+https://github.com/eightmm/Featurizer.git
```

### Install for Development

```bash
git clone https://github.com/eightmm/Featurizer.git
cd Featurizer
pip install -e .
```

### Dependencies

```bash
# Core dependencies
pip install torch numpy pandas rdkit-pypi

# For protein features
pip install freesasa biopython

# For graph features (optional)
pip install dgl
```

## 🚀 Quick Start

### Molecule Features

```python
from featurizer.molecule_featurizer import MoleculeFeaturizer
from rdkit import Chem

# Initialize featurizer
featurizer = MoleculeFeaturizer()

# Method 1: From RDKit mol object (recommended for 3D structures)
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
features = featurizer.get_feature(mol)

# Method 2: From SMILES string
features = featurizer.get_feature("CC(=O)Oc1ccccc1C(=O)O")

# Get graph representation for GNNs
node, edge = featurizer.get_graph(mol)  # Returns (node, edge) tuple
```

### Protein Features

```python
from featurizer.protein_featurizer import ProteinFeaturizer

# Initialize featurizer
featurizer = ProteinFeaturizer()

# Extract features from PDB file
features = featurizer.extract("protein.pdb")

# Access node and edge features
node_features = features['node']  # Per-residue features
edge_features = features['edge']  # Residue-residue interactions
```

## 📊 Feature Details

### Molecule Features

#### `get_feature()` Returns:
```python
{
    'descriptor': torch.Tensor,      # Shape: [20] - Molecular descriptors
    'maccs': torch.Tensor,           # Shape: [167] - MACCS keys
    'morgan': torch.Tensor,          # Shape: [2048] - Morgan fingerprint
    'morgan_count': torch.Tensor,    # Shape: [2048] - Morgan count fingerprint
    'feature_morgan': torch.Tensor,  # Shape: [2048] - Feature Morgan fingerprint
    'rdkit': torch.Tensor,           # Shape: [2048] - RDKit fingerprint
    'atom_pair': torch.Tensor,       # Shape: [2048] - Atom pair fingerprint
    'topological_torsion': torch.Tensor,  # Shape: [2048] - Topological torsion
    'pharmacophore2d': torch.Tensor  # Shape: [1024] - 2D pharmacophore
}
```

**Molecular Descriptors (20 features):**
- **Physicochemical (12)**: MW, LogP, TPSA, rotatable bonds, flexibility, HBD, HBA, n_atoms, n_bonds, n_rings, n_aromatic_rings, heteroatom_ratio
- **Drug-likeness (5)**: Lipinski violations, passes Lipinski, QED, heavy atoms, fraction sp3
- **Structural (3)**: Ring systems, max ring size, average ring size

#### `get_graph()` Returns:
Returns a tuple of (node, edge) dictionaries:

```python
node, edge = featurizer.get_graph(mol)

# Node dictionary contains:
node = {
    'coords': torch.Tensor,      # Shape: [n_atoms, 3] - 3D coordinates
    'node_feats': torch.Tensor    # Shape: [n_atoms, 122] - Atom features
}

# Edge dictionary contains:
edge = {
    'edges': torch.Tensor,        # Shape: [2, n_edges] - Source/destination indices
    'edge_feats': torch.Tensor    # Shape: [n_edges, 44] - Bond features
}
```

**Node Features (122 dimensions):**
- Atom type (one-hot for H, C, N, O, S, P, F, Cl, Br, I, UNK)
- Period and group in periodic table
- Aromaticity, ring membership
- Formal charge, electronegativity
- Degree features and hybridization
- Ring properties
- SMARTS pattern matches (H-bond donor/acceptor, hydrophobic)
- Stereochemistry
- Partial charges
- Extended neighborhood features

**Edge Features (44 dimensions):**
- Bond type (single, double, triple, aromatic)
- Bond stereochemistry
- Ring membership, conjugation
- Rotatability
- Degree-based features
- Ring properties

### Protein Features

#### Node Features (Per Residue):
- Residue type (20 amino acids + UNK)
- Terminal flags (N-terminal, C-terminal)
- Geometric features (distances, angles, dihedrals)
- SASA (Solvent Accessible Surface Area)
- Secondary structure elements
- Local coordinate frames

#### Edge Features (Residue Pairs):
- Spatial distances (CA-CA, SC-SC, CA-SC, SC-CA)
- Relative position encoding
- 3D orientation vectors
- Contact indicators

## 🔧 Advanced Usage

### Working with 3D Structures

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Create molecule with 3D coordinates
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
AllChem.EmbedMolecule(mol)  # Generate 3D structure
AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry

# Featurizer preserves 3D coordinates
featurizer = MoleculeFeaturizer()
node, edge = featurizer.get_graph(mol, add_hs=True)  # Adds H with proper 3D coords
coords = node['coords']  # 3D coordinates preserved
```

### Custom Feature Extraction

```python
# Direct access to specific feature methods
from featurizer.molecule_featurizer.molecule_feature import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()
mol = Chem.MolFromSmiles("CCO")

# Get individual feature types
phys_features = featurizer.get_physicochemical_features(mol)
drug_features = featurizer.get_druglike_features(mol)
struct_features = featurizer.get_structural_features(mol)
fingerprints = featurizer.get_fingerprints(mol)
```

### Batch Processing

```python
# Process multiple molecules
molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
features_list = []

for smiles in molecules:
    features = featurizer.get_feature(smiles)
    features_list.append(features)

# Process multiple proteins
pdb_files = ["1abc.pdb", "2def.pdb", "3ghi.pdb"]
protein_featurizer = ProteinFeaturizer()

for pdb_file in pdb_files:
    features = protein_featurizer.extract(pdb_file)
    # Save or process features
```

### Integration with Graph Neural Networks

```python
import dgl
import torch.nn as nn

# Get molecular graph
node, edge = featurizer.get_graph(mol)

# Create DGL graph for GNN
# Extract source and destination indices from edges tensor
src = edge['edges'][0]  # First row contains source indices
dst = edge['edges'][1]  # Second row contains destination indices
g = dgl.graph((src, dst))

# Add features to graph
g.ndata['feat'] = node['node_feats']
g.ndata['coords'] = node['coords']
g.edata['feat'] = edge['edge_feats']

# Use with your GNN model
# model = YourGNNModel(in_dim=122, hidden_dim=128, out_dim=1)
# output = model(g, g.ndata['feat'], g.edata['feat'])
```

## 📋 Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.9.0
- RDKit ≥ 2020.09
- NumPy ≥ 1.19.0
- Pandas ≥ 1.3.0
- DGL ≥ 0.9.0 (optional, for graph features)
- FreeSASA ≥ 2.1.0 (for protein SASA)
- BioPython ≥ 1.79 (for protein processing)

## 🏗️ Project Structure

```
featurizer/
├── molecule_featurizer/
│   ├── __init__.py
│   └── molecule_feature.py    # Core molecule featurization
├── protein_featurizer/
│   ├── __init__.py
│   └── protein_feature.py     # Core protein featurization
└── __init__.py                 # Package exports
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{featurizer2025,
  title = {Featurizer: A unified framework for molecular and protein feature extraction},
  author = {Jaemin Sim},
  year = {2025},
  url = {https://github.com/eightmm/Featurizer}
}
```

## 🐛 Issues and Support

For bugs and feature requests, please use the [GitHub Issues](https://github.com/eightmm/Featurizer/issues) page.

## 📊 Version History

- **v1.0.0** (2025-01): Major refactoring
  - Unified MoleculeFeaturizer class
  - Enhanced 3D coordinate support
  - Improved graph feature generation
  - Removed domain-specific features for generalization

---

Made with ❤️ for the computational chemistry and bioinformatics community
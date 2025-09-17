# Featurizer

A comprehensive Python package for extracting features from both **molecules** and **proteins** for machine learning applications, with special support for graph neural networks.

## 🌟 Key Features

### 🧪 Molecule Featurizer
- **40 molecular descriptors** covering physicochemical, topological, and structural properties
- **9 types of molecular fingerprints** (Morgan, MACCS, RDKit, etc.)
- **Graph representations** with 3D coordinate preservation for GNNs
- **Dual input support**: RDKit mol objects or SMILES strings

### 🧬 Protein Featurizer
- **PDB file processing** with automatic standardization
- **Residue-level features** including geometry, SASA, and contacts
- **Graph representations** for protein structure networks
- **Multiple feature extraction methods** for different ML applications

## 📦 Installation

```bash
# Install from GitHub
pip install git+https://github.com/eightmm/Featurizer.git

# For development
git clone https://github.com/eightmm/Featurizer.git
cd Featurizer
pip install -e .
```

### Dependencies
```bash
pip install torch numpy pandas rdkit-pypi  # Core
pip install freesasa biopython            # For proteins
pip install dgl                           # For GNNs (optional)
```

## 🚀 Quick Start

### Molecule Features
```python
from featurizer import MoleculeFeaturizer

# Create featurizer
featurizer = MoleculeFeaturizer()

# Extract all features
features = featurizer.get_feature("CCO")  # ethanol
# Returns dict with 'descriptor' (40D), 'morgan' (2048D), 'maccs' (167D), etc.

# Get graph representation for GNNs
node, edge = featurizer.get_graph("c1ccccc1")  # benzene
# node['coords']: 3D coordinates, node['node_feats']: atom features
# edge['edges']: connectivity, edge['edge_feats']: bond features
```

### Protein Features
```python
from featurizer import ProteinFeaturizer

# Create featurizer
featurizer = ProteinFeaturizer("protein.pdb")

# Extract all features at once
features = featurizer.get_all_features()
# node_features: per-residue features
# edge_features: residue-residue interactions

# Or extract specific features
sasa = featurizer.get_sasa_features()       # Solvent accessibility
contacts = featurizer.get_contact_map(8.0)  # Contact map at 8Å cutoff
```

## 📊 Feature Overview

### Molecules
- **Descriptors**: 40 normalized molecular properties → [Details](docs/molecular_descriptors.md)
- **Fingerprints**: 9 types including Morgan, MACCS, RDKit → [Details](docs/feature_types.md#fingerprints)
- **Graph Features**: 122D atom features, 44D bond features → [Details](docs/feature_types.md#graph-representations)

### Proteins
- **Node Features**: Residue type, geometry, SASA, secondary structure
- **Edge Features**: Distances, orientations, contacts
- **Sequence Features**: AAC, DPC, CTD descriptors → [Details](docs/feature_types.md#protein-features)

## 🔧 Advanced Examples

### Batch Processing
```python
# Process multiple molecules efficiently
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
features = [featurizer.get_feature(smi) for smi in smiles_list]

# Stack for neural network input
import torch
descriptors = torch.stack([f['descriptor'] for f in features])
```

### GNN Integration
```python
import dgl

# Get molecular graph
node, edge = featurizer.get_graph(mol)

# Create DGL graph
g = dgl.graph((edge['edges'][0], edge['edges'][1]))
g.ndata['feat'] = node['node_feats']
g.edata['feat'] = edge['edge_feats']
```

### 3D Structure Preservation
```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Generate 3D coordinates
mol = Chem.MolFromSmiles("CCO")
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

# Extract with 3D coords preserved
node, edge = featurizer.get_graph(mol)
coords_3d = node['coords']  # [n_atoms, 3]
```

## 📖 Documentation

- [Molecular Descriptors Reference](docs/molecular_descriptors.md)
- [Feature Types Guide](docs/feature_types.md)
- [API Examples](examples/)

## 📋 Requirements

- Python ≥ 3.7
- PyTorch ≥ 2.0.0
- RDKit ≥ 2023.03
- NumPy ≥ 1.20.0
- See [requirements.txt](requirements.txt) for full list

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 📖 Citation

```bibtex
@software{featurizer2025,
  title = {Featurizer: Unified molecular and protein feature extraction},
  author = {Jaemin Sim},
  year = {2025},
  url = {https://github.com/eightmm/Featurizer}
}
```

## 🐛 Support

For issues and questions, please use [GitHub Issues](https://github.com/eightmm/Featurizer/issues).

---
Made with ❤️ for the computational chemistry and bioinformatics community
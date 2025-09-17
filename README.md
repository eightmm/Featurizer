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
pip install git+https://github.com/eightmm/Featurizer.git
```

## 🚀 Quick Start

### Molecule Features
```python
from featurizer import MoleculeFeaturizer
from rdkit import Chem

featurizer = MoleculeFeaturizer()

# From SDF file
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    features = featurizer.get_feature(mol)
    node, edge = featurizer.get_graph(mol)

# From SMILES
smiles = "CC(=O)Oc1ccccc1C(=O)O"
features = featurizer.get_feature(smiles)
node, edge = featurizer.get_graph(smiles)
```

### Protein Features
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

features = featurizer.get_all_features()
sasa = featurizer.get_sasa_features()
contacts = featurizer.get_contact_map(8.0)
```

## 📊 Feature Overview

### Molecules
- **Descriptors**: 40 normalized molecular properties → [Details](docs/molecular_descriptors.md)
- **Fingerprints**: 9 types including Morgan, MACCS, RDKit → [Details](docs/feature_types.md#fingerprints)
- **Graph Features**: 122D atom features, 44D bond features → [Details](docs/feature_types.md#graph-representations)

### Proteins
- **Node Features**: Residue type, geometry, SASA, secondary structure
- **Edge Features**: Distances, orientations, contacts
- **Graph Representations**: Residue-residue interaction networks → [Details](docs/feature_types.md#protein-features)

## 🔧 Advanced Examples

### Batch Processing
```python
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
features = [featurizer.get_feature(smi) for smi in smiles_list]

import torch
descriptors = torch.stack([f['descriptor'] for f in features])
```

### GNN Integration
```python
import dgl

node, edge = featurizer.get_graph(mol)

g = dgl.graph((edge['edges'][0], edge['edges'][1]))
g.ndata['feat'] = node['node_feats']
g.edata['feat'] = edge['edge_feats']
```

### 3D Structure Preservation
```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

node, edge = featurizer.get_graph(mol)
coords_3d = node['coords']
```

## 📖 Documentation

- [Molecular Descriptors Reference](docs/molecular_descriptors.md)
- [Feature Types Guide](docs/feature_types.md)
- [API Examples](examples/)

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
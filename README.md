# Featurizer

A comprehensive Python package for extracting features from both **molecules** and **proteins** for machine learning applications, with special support for graph neural networks.


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

# Atom-level features (node/edge format with SASA included)
atom_node, atom_edge = featurizer.get_atom_features(distance_cutoff=4.0)
# atom_node contains: coord, atom_tokens, sasa, residue_token, atom_element

# Residue-level features (node/edge format)
res_node, res_edge = featurizer.get_residue_features(distance_cutoff=8.0)
```

## 📊 Feature Overview

### Molecules
- **Descriptors**: 40 normalized molecular properties → [Details](docs/molecular_descriptors.md)
- **Fingerprints**: 9 types including Morgan, MACCS, RDKit → [Details](docs/molecule_feature.md)
- **Graph Features**: 122D atom features, 44D bond features → [Details](docs/molecule_graph.md)

### Proteins
- **Atom Features**: 175 token types with atomic SASA → [Details](docs/protein_atom_feature.md)
- **Residue Features**: Geometry, SASA, contacts, secondary structure → [Details](docs/protein_residue_feature.md)
- **Graph Representations**: Both atom and residue-level networks

## 🔧 Advanced Examples

### Contact Maps with Different Thresholds
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Different thresholds for different analyses
close_contacts = featurizer.get_contact_map(cutoff=4.5)   # Close contacts only
standard_contacts = featurizer.get_contact_map(cutoff=8.0) # Standard threshold
extended_contacts = featurizer.get_contact_map(cutoff=12.0) # Extended interactions

# Access contact information
edges = standard_contacts['edges']
distances = standard_contacts['distances']
adjacency = standard_contacts['adjacency_matrix']
```

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

- **[Feature Types Overview](docs/feature_types.md)** - Quick overview of all features
- **[Molecular Descriptors & Fingerprints](docs/molecule_feature.md)** - Molecular features guide
- **[Molecule Graph Features](docs/molecule_graph.md)** - Graph representations for molecules
- **[Protein Residue Features](docs/protein_residue_feature.md)** - Residue-level features guide
- **[Protein Atom Features](docs/protein_atom_feature.md)** - Atom-level features guide
- **[Molecular Descriptors Reference](docs/molecular_descriptors.md)** - Complete descriptor reference
- **[Examples](examples/)** - Code examples and tutorials

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

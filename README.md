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

# From SDF file
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    featurizer = MoleculeFeaturizer(mol)  # Initialize with molecule
    features = featurizer.get_feature()
    node, edge = featurizer.get_graph()

# From SMILES
featurizer = MoleculeFeaturizer("CC(=O)Oc1ccccc1C(=O)O")
features = featurizer.get_feature()  # All descriptors and fingerprints
node, edge = featurizer.get_graph()  # Graph representation
```

### Protein Features
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Atom-level features (node/edge format with SASA included)
atom_node, atom_edge = featurizer.get_atom_features(distance_cutoff=4.0)

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

### Custom SMARTS Patterns for Molecules
```python
from featurizer import MoleculeFeaturizer

# Define custom SMARTS patterns
custom_patterns = {
    'aromatic_nitrogen': 'n',
    'carboxyl': 'C(=O)O',
    'hydroxyl': '[OH]',
    'amine': '[NX3;H2,H1;!$(NC=O)]'
}

# Initialize with custom patterns
featurizer = MoleculeFeaturizer("c1ccncc1CCO", custom_smarts=custom_patterns)

# Get graph with custom features included
node, edge = featurizer.get_graph()
# node['node_feats'] now includes custom SMARTS matches
# node['custom_smarts_feats'] contains just the custom features
# node['custom_smarts_names'] contains pattern names

# Or get custom features separately
custom_feats = featurizer.get_custom_smarts_features()
# Returns: {'features': tensor, 'names': [...], 'patterns': {...}}
```

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
from featurizer import MoleculeFeaturizer
import torch

smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
all_features = []

for smiles in smiles_list:
    featurizer = MoleculeFeaturizer(smiles)
    features = featurizer.get_feature()
    all_features.append(features['descriptor'])

descriptors = torch.stack(all_features)
```


## 📖 Documentation

- **[Feature Types Overview](docs/feature_types.md)** - Quick overview of all features
- **[Molecular Descriptors & Fingerprints](docs/molecule_feature.md)** - Molecular features guide
- **[Molecule Graph Features](docs/molecule_graph.md)** - Graph representations for molecules
- **[Protein Residue Features](docs/protein_residue_feature.md)** - Residue-level features guide
- **[Protein Atom Features](docs/protein_atom_feature.md)** - Atom-level features guide
- **[Molecular Descriptors Reference](docs/molecular_descriptors.md)** - Complete descriptor reference
- **[Examples](examples/)** - Code examples and tutorials

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


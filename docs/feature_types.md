# Feature Types Overview

The Featurizer package provides comprehensive feature extraction for both molecules and proteins. This document provides a high-level overview of available features.

## 📚 Detailed Documentation

- **[Molecule Features](molecule_features.md)** - Complete guide for molecular feature extraction
- **[Protein Features](protein_features.md)** - Complete guide for protein feature extraction
- **[Molecular Descriptors](molecular_descriptors.md)** - Detailed reference for 40 molecular descriptors

## 🧪 Molecule Features Summary

### Available Features
- **40 Molecular Descriptors**: Physicochemical, topological, and structural properties
- **9 Fingerprint Types**: Morgan, MACCS, RDKit, Atom Pair, etc.
- **Graph Representations**: 122D atom features, 44D bond features with 3D coordinates

### Quick Example
```python
from featurizer import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()
features = featurizer.get_feature("CCO")  # All features
node, edge = featurizer.get_graph("CCO")  # Graph representation
```

➡️ See [Molecule Features Documentation](molecule_features.md) for complete details

## 🧬 Protein Features Summary

### Available Features
- **Sequence Features**: Residue types and one-hot encoding
- **Geometric Features**: Dihedrals, curvature, torsion, distances
- **SASA Features**: 10-component solvent accessibility analysis
- **Contact Maps**: Customizable distance thresholds (4.5-12.0 Å)
- **Graph Representations**: Node and edge features for protein networks

### Quick Example
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")
features = featurizer.get_all_features()  # All features
contacts = featurizer.get_contact_map(cutoff=8.0)  # Contact map
```

➡️ See [Protein Features Documentation](protein_features.md) for complete details

## 🔧 Common Use Cases

### Drug Discovery
- Molecular descriptors for ADMET prediction
- Morgan fingerprints for similarity searching
- Protein-ligand interaction features

### Structural Biology
- Protein contact maps for fold recognition
- SASA for binding site identification
- Geometric features for structure validation

### Machine Learning
- Graph features for GNNs
- Descriptors for traditional ML (RF, XGBoost)
- Fingerprints for clustering and classification

## 🚀 Quick Start Guide

### Installation
```bash
pip install git+https://github.com/eightmm/Featurizer.git
```

### Basic Usage
```python
# Molecules
from featurizer import MoleculeFeaturizer
mol_feat = MoleculeFeaturizer()
mol_features = mol_feat.get_feature("c1ccccc1")

# Proteins
from featurizer import ProteinFeaturizer
prot_feat = ProteinFeaturizer("structure.pdb")
prot_features = prot_feat.get_all_features()
```

## 📊 Performance Guidelines

### Molecules
- **Descriptors**: ~1ms per molecule
- **All fingerprints**: ~5ms per molecule
- **Graph features**: ~10ms per molecule

### Proteins
- **Small (<100 residues)**: ~100ms total
- **Medium (100-500 residues)**: ~200-300ms
- **Large (>500 residues)**: ~500ms+

## 📖 Further Reading

- [API Reference](../README.md)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/eightmm/Featurizer)
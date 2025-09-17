# Feature Types Overview

The Featurizer package provides comprehensive feature extraction for both molecules and proteins. This document provides a high-level overview with links to detailed documentation.

## 📚 Documentation Structure

### 🧪 Molecule Features
- **[Molecular Descriptors & Fingerprints](molecule_feature.md)** - 40 descriptors and 9 fingerprint types
- **[Molecule Graph Representations](molecule_graph.md)** - Graph features for GNNs
- **[Molecular Descriptors Reference](molecular_descriptors.md)** - Detailed descriptor documentation

### 🧬 Protein Features
- **[Residue-Level Features](protein_residue_feature.md)** - Comprehensive residue-based analysis
- **[Atom-Level Features](protein_atom_feature.md)** - 175 token types with atomic SASA

## 🧪 Molecule Features Summary

### Available Features
- **40 Molecular Descriptors**: Physicochemical, topological, and structural properties
- **9 Fingerprint Types**: Morgan, MACCS, RDKit, Atom Pair, etc.
- **Graph Representations**: 122D atom features, 44D bond features with 3D coordinates

### Quick Example
```python
from featurizer import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()

# Descriptors and fingerprints
features = featurizer.get_feature("CCO")

# Graph representation
node, edge = featurizer.get_graph("CCO")
```

**Learn More:**
- ➡️ [Descriptors & Fingerprints](molecule_feature.md) - Complete molecular features
- ➡️ [Graph Features](molecule_graph.md) - GNN-ready representations

## 🧬 Protein Features Summary

### Available Features

#### Atom-Level
- **175 Token Types**: Unique residue-atom combinations
- **Atomic SASA**: Solvent accessible surface area per atom
- **3D Coordinates**: Precise atomic positions

#### Residue-Level
- **Sequence Features**: Residue types and one-hot encoding
- **Geometric Features**: Dihedrals, curvature, torsion, distances
- **SASA Features**: 10-component solvent accessibility analysis
- **Contact Maps**: Customizable distance thresholds (4.5-12.0 Å)
- **Graph Representations**: Node and edge features for protein networks

### Quick Example
```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Atom-level features
atom_features = featurizer.get_atom_features_with_sasa()

# Residue-level features
node, edge = featurizer.get_features()
contacts = featurizer.get_contact_map(cutoff=8.0)
```

**Learn More:**
- ➡️ [Residue Features](protein_residue_feature.md) - Residue-level analysis
- ➡️ [Atom Features](protein_atom_feature.md) - Atomic-level tokenization

## 🔧 Common Use Cases

### Drug Discovery
- Molecular descriptors for ADMET prediction
- Morgan fingerprints for similarity searching
- Protein-ligand interaction features
- Binding site identification with atomic SASA

### Structural Biology
- Protein contact maps for fold recognition
- Atomic SASA for binding site identification
- Geometric features for structure validation
- Atom-level analysis for quality assessment

### Machine Learning
- Graph features for GNNs (molecules and proteins)
- Descriptors for traditional ML (RF, XGBoost)
- Fingerprints for clustering and classification
- Atom tokens for transformer models

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
mol_graph = mol_feat.get_graph("c1ccccc1")

# Proteins (Residue-level)
from featurizer import ProteinFeaturizer
prot_feat = ProteinFeaturizer("structure.pdb")
node, edge = prot_feat.get_features()

# Proteins (Atom-level)
atom_features = prot_feat.get_atom_features_with_sasa()
```

## 📊 Performance Guidelines

### Molecules
- **Descriptors**: ~1ms per molecule
- **All fingerprints**: ~5ms per molecule
- **Graph features**: ~10ms per molecule

### Proteins
- **Atom tokenization**: ~10-50ms
- **Atomic SASA**: ~100-300ms
- **Residue features**: ~100-500ms total
- **Small (<100 residues)**: ~100ms
- **Medium (100-500 residues)**: ~200-300ms
- **Large (>500 residues)**: ~500ms+

## 📖 Complete Documentation

### Detailed Guides
- **[Molecular Descriptors & Fingerprints](molecule_feature.md)**
- **[Molecule Graph Representations](molecule_graph.md)**
- **[Molecular Descriptors Reference](molecular_descriptors.md)**
- **[Protein Residue Features](protein_residue_feature.md)**
- **[Protein Atom Features](protein_atom_feature.md)**

### Additional Resources
- [API Reference](../README.md)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/eightmm/Featurizer)
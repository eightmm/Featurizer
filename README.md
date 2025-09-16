# Protein Featurizer

A Python tool for extracting structural features from protein PDB files for machine learning applications.

## Features

- PDB file standardization and cleaning
- Residue-level feature extraction
- Geometric features (distances, angles, dihedrals)
- Solvent accessible surface area (SASA) calculation
- Graph-based interaction features
- Support for multi-chain proteins

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Basic usage
python feature.py input.pdb

# With PDB standardization
python feature.py input.pdb --standardize

# Specify output file
python feature.py input.pdb -o features.pt --standardize
```

### Python API

```python
from feature import ResidueFeaturizer

# Initialize featurizer with PDB file
featurizer = ResidueFeaturizer('protein.pdb')

# Extract features
node_features, edge_features = featurizer.get_features()

# Access specific features
residues = featurizer.get_residue()
sasa = featurizer.get_SASA()
```

## Feature Types

### Node Features (per residue)
- Residue type (one-hot encoding)
- Terminal flags (N-terminal, C-terminal)
- Self-interaction distances and vectors
- Dihedral angles (backbone and side-chain)
- Backbone curvature and torsion
- Solvent accessible surface area (SASA)
- Forward/reverse residue connections

### Edge Features (residue pairs)
- Interaction distances (CA-CA, SC-SC, CA-SC, SC-CA)
- Relative position encoding
- Interaction vectors

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- FreeSASA

## License

MIT
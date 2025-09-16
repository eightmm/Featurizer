# Protein Featurizer

A comprehensive Python toolkit for extracting structural features from protein PDB files for machine learning applications.

## Features

- **PDB Standardization**: Clean and standardize PDB files (remove waters, DNA/RNA, reorder atoms)
- **Feature Extraction**: Comprehensive residue-level and interaction features
- **Geometric Analysis**: Distances, angles, dihedrals, curvature, and torsion
- **SASA Calculation**: Solvent accessible surface area analysis
- **Graph Representation**: Protein structure as graph with node and edge features
- **Batch Processing**: Process multiple PDB files efficiently
- **Modular Design**: Separate modules for different functionalities

## Project Structure

```
protein-featurizer/
├── pdb_standardizer.py   # PDB cleaning and standardization
├── residue_featurizer.py # Feature extraction from proteins
├── main.py               # Main pipeline orchestrator
├── feature.py            # Legacy all-in-one script
├── requirements.txt      # Package dependencies
└── README.md            # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Single File Processing

```bash
# Process a single PDB file with standardization
python main.py protein.pdb -o features.pt

# Process without standardization
python main.py protein.pdb -o features.pt --no-standardize

# Keep hydrogen atoms during standardization
python main.py protein.pdb -o features.pt --keep-hydrogens
```

#### Batch Processing

```bash
# Process all PDB files in a directory
python main.py --batch input_dir/ output_dir/

# Process with specific pattern
python main.py --batch input_dir/ output_dir/ --pattern "**/*_protein.pdb"

# Reprocess existing files
python main.py --batch input_dir/ output_dir/ --no-skip
```

#### Module-Specific Usage

```bash
# Standardize PDB only
python pdb_standardizer.py input.pdb output_clean.pdb

# Extract features only (requires clean PDB)
python residue_featurizer.py clean.pdb -o features.pt
```

### Python API

#### Complete Pipeline

```python
from pdb_standardizer import PDBStandardizer
from residue_featurizer import ResidueFeaturizer

# Step 1: Standardize PDB
standardizer = PDBStandardizer(remove_hydrogens=True)
clean_pdb = standardizer.standardize('input.pdb', 'clean.pdb')

# Step 2: Extract features
featurizer = ResidueFeaturizer(clean_pdb)
node_features, edge_features = featurizer.get_features()

# Save features
import torch
torch.save({'node': node_features, 'edge': edge_features}, 'features.pt')
```

#### Direct Feature Extraction

```python
from residue_featurizer import ResidueFeaturizer

# Initialize with PDB file
featurizer = ResidueFeaturizer('protein.pdb')

# Get specific features
residues = featurizer.get_residues()
sasa = featurizer.calculate_sasa()
terminal_flags = featurizer.get_terminal_flags()

# Get all features
node_features, edge_features = featurizer.get_features()
```

## Feature Types

### Node Features (Per Residue)

- **Residue Identity**: One-hot encoding of amino acid type (21 classes)
- **Terminal Flags**: N-terminal and C-terminal indicators
- **Geometric Features**:
  - Self-interaction distances within residue
  - Backbone dihedral angles (φ, ψ, ω)
  - Side-chain dihedral angles (χ1-χ5)
  - Backbone curvature and torsion
- **SASA Features**: Total, polar, apolar, main chain, side chain (absolute and relative)
- **Local Coordinate Frames**: Residue-specific coordinate system
- **Sequential Connections**: Forward/reverse residue vectors and distances

### Edge Features (Residue Pairs)

- **Interaction Distances**: CA-CA, SC-SC, CA-SC, SC-CA distances
- **Relative Position**: Sequential distance encoding (one-hot)
- **Interaction Vectors**: 3D vectors between residue pairs

### Output Format

Features are saved as PyTorch tensors in a dictionary:

```python
{
    'node': {
        'coord': Tensor[N, 2, 3],  # CA and SC coordinates
        'node_scalar_features': Tuple of scalar feature tensors,
        'node_vector_features': Tuple of vector feature tensors
    },
    'edge': {
        'edges': Tuple[Tensor, Tensor],  # Source and destination indices
        'edge_scalar_features': Tuple of scalar feature tensors,
        'edge_vector_features': Tuple of vector feature tensors
    },
    'metadata': {
        'input_file': str,
        'standardized': bool,
        'hydrogens_removed': bool
    }
}
```

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- FreeSASA >= 2.1.0

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{protein_featurizer,
  title = {Protein Featurizer: A toolkit for protein structure feature extraction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/eightmm/protein-featurizer}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
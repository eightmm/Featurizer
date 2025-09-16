# Protein Featurizer

A Python package for extracting structural features from protein PDB files for machine learning applications.

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/eightmm/protein-featurizer.git
```

### Install for Development

```bash
git clone https://github.com/eightmm/protein-featurizer.git
cd protein-featurizer
pip install -e .
```

## Quick Start

```python
from protein_featurizer import Featurizer

# Initialize featurizer
featurizer = Featurizer()

# Extract features from a PDB file
features = featurizer.extract("protein.pdb")

# Save features to file
features = featurizer.extract("protein.pdb", save_to="features.pt")

# Process multiple files
results = featurizer.extract_batch(
    ["protein1.pdb", "protein2.pdb"],
    output_dir="features/"
)
```

## Features

- 🧬 **PDB Standardization**: Automatic cleaning and standardization of PDB files
- 📊 **Comprehensive Features**: Geometric, chemical, and interaction features
- 🚀 **Simple API**: Easy-to-use interface for both beginners and experts
- 📦 **Batch Processing**: Efficiently process multiple proteins
- 🔧 **Modular Design**: Use individual components as needed

## Usage Examples

### Basic Usage

```python
from protein_featurizer import Featurizer

# Create featurizer with default settings
featurizer = Featurizer()

# Extract features
features = featurizer.extract("protein.pdb")

# Access node and edge features
node_features = features['node']
edge_features = features['edge']
```

### Advanced Usage

```python
from protein_featurizer import Featurizer

# Custom configuration
featurizer = Featurizer(
    standardize=True,      # Clean PDB file first
    keep_hydrogens=False   # Remove hydrogen atoms
)

# Process and save
features = featurizer.extract(
    "protein.pdb",
    save_to="features.pt"
)

# Batch processing
pdb_files = ["1abc.pdb", "2def.pdb", "3ghi.pdb"]
results = featurizer.extract_batch(
    pdb_files,
    output_dir="processed_features/",
    skip_existing=True,
    verbose=True
)
```

### Using Individual Components

```python
from protein_featurizer import PDBStandardizer, ResidueFeaturizer

# Step 1: Standardize PDB
standardizer = PDBStandardizer(remove_hydrogens=True)
clean_pdb = standardizer.standardize("input.pdb", "clean.pdb")

# Step 2: Extract features
featurizer = ResidueFeaturizer(clean_pdb)
node_features, edge_features = featurizer.get_features()
```

### Command Line Interface

```bash
# Extract features
protein-featurizer protein.pdb -o features.pt

# Batch processing
protein-featurizer --batch input_dir/ output_dir/

# Just standardize PDB
pdb-standardize input.pdb output_clean.pdb
```

## Extracted Features

### Node Features (Per Residue)
- Residue type (one-hot encoding)
- Terminal flags (N/C-terminal)
- Geometric features (distances, angles, dihedrals)
- SASA (Solvent Accessible Surface Area)
- Local coordinate frames
- Sequential connections

### Edge Features (Residue Pairs)
- Interaction distances (CA-CA, SC-SC, CA-SC, SC-CA)
- Relative position encoding
- 3D interaction vectors

## API Reference

### `Featurizer` Class

The main API for feature extraction.

#### Methods

- `extract(pdb_file, save_to=None)`: Extract features from a single PDB file
- `extract_batch(pdb_files, output_dir=None, skip_existing=True, verbose=True)`: Process multiple files
- `from_clean_pdb(pdb_file)`: Extract from pre-cleaned PDB (class method)
- `standardize_only(input_pdb, output_pdb, keep_hydrogens=False)`: Only standardize PDB (static method)

### Output Format

```python
{
    'node': {
        'coord': Tensor[N, 2, 3],  # CA and SC coordinates
        'node_scalar_features': tuple,
        'node_vector_features': tuple
    },
    'edge': {
        'edges': (src_indices, dst_indices),
        'edge_scalar_features': tuple,
        'edge_vector_features': tuple
    },
    'metadata': {
        'input_file': str,
        'standardized': bool,
        'hydrogens_removed': bool
    }
}
```

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- Pandas ≥ 1.3.0
- FreeSASA ≥ 2.1.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{protein_featurizer,
  title = {Protein Featurizer: A Python package for protein structure feature extraction},
  author = {Jaemin Sim},
  year = {2025},
  url = {https://github.com/eightmm/protein-featurizer}
}
```

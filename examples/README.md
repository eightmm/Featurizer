# Examples

This directory contains example scripts demonstrating how to use the Protein Featurizer package.

## Available Examples

### 1. `basic_usage.py`
- Simple feature extraction from a single PDB file
- Saving features to disk
- Using pre-cleaned PDB files

### 2. `batch_processing.py`
- Processing multiple PDB files
- Directory-based batch processing
- Error handling in batch operations
- Processing with different settings

### 3. `advanced_usage.py`
- Using individual components (PDBStandardizer, ResidueFeaturizer)
- Custom processing pipelines
- Feature analysis and statistics
- Integration with machine learning frameworks
- Memory-efficient processing for large datasets

## Running the Examples

First, install the package:
```bash
pip install git+https://github.com/eightmm/protein-featurizer.git
```

Then run any example:
```bash
python examples/basic_usage.py
python examples/batch_processing.py
python examples/advanced_usage.py
```

## Note

These examples use placeholder PDB file names. Replace them with your actual PDB files:
- Replace `"protein.pdb"` with your PDB file path
- Replace `"path/to/pdb/files"` with your actual directory paths

## Quick Start

The simplest way to extract features:

```python
from protein_featurizer import Featurizer

# Extract features from a PDB file
featurizer = Featurizer()
features = featurizer.extract("your_protein.pdb")

# Access the features
node_features = features['node']
edge_features = features['edge']
```
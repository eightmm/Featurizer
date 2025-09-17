# Featurizer

A comprehensive Python package for extracting features from both **molecule** (small molecule/drug) and **protein** structures for machine learning applications.

## Features

### 🧪 Molecule Features (Small Molecules/Drugs)
- Extract physicochemical, structural, and fingerprint features
- Support for SMILES strings and RDKit mol objects
- Universal descriptors applicable to any molecule
- Multiple fingerprint types (Morgan, MACCS, RDKit, etc.)

### 🧬 Protein Features (Macromolecules)
- Extract geometric, chemical, and interaction features from PDB files
- Residue-based node and edge features
- Automatic PDB standardization
- SASA and structural feature calculations

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/eightmm/featurizer.git
```

### Install for Development

```bash
git clone https://github.com/eightmm/featurizer.git
cd featurizer
pip install -e .
```

## Quick Start

### Molecule Features (Small Molecules)

```python
from featurizer import MoleculeFeaturizer
from rdkit import Chem

# Initialize featurizer
featurizer = MoleculeFeaturizer()

# From SMILES string
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
features = featurizer.extract(smiles)

# From RDKit mol object
mol = Chem.MolFromSmiles(smiles)
features = featurizer.extract(mol)

# Access different feature types
descriptors = features['descriptors']  # Physicochemical descriptors
fingerprints = features['fingerprints']  # Various fingerprints (Morgan, MACCS, etc.)
metadata = features['metadata']  # Input information
```

### Protein Features (Macromolecules)

```python
from featurizer import ProteinFeaturizer

# Initialize featurizer
featurizer = ProteinFeaturizer()

# Extract features from a PDB file
features = featurizer.extract("protein.pdb")

# Access node and edge features
node_features = features['node']
edge_features = features['edge']

# Save features to file
features = featurizer.extract("protein.pdb", save_to="features.pt")
```

## Feature Dimensions

### Molecule Features (Small Molecules)

#### Node Features (Per Atom)
- **node_scalar_features**: `[num_atoms, 26]`
  - Atom type one-hot encoding: 13 dims (C, N, O, S, P, F, Cl, Br, I, B, Si, Se, UNK)
  - Atomic number (normalized): 1 dim
  - Degree (normalized): 1 dim
  - Formal charge (normalized): 1 dim
  - Hybridization one-hot: 6 dims (SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED)
  - Is aromatic: 1 dim
  - Is in ring: 1 dim
  - Number of hydrogens (normalized): 1 dim
  - Implicit valence (normalized): 1 dim
  - Mass (normalized): 1 dim
- **node_vector_features**: `[num_atoms, 1]`
  - Partial charge (Gasteiger): 1 dim
- **coordinates**: `[num_atoms, 3]` (x, y, z) - 2D placeholder or 3D if computed

#### Edge Features (Per Bond)
- **edge_scalar_features**: `[num_edges, 7]`
  - Bond type one-hot: 4 dims (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  - Is conjugated: 1 dim
  - Is in ring: 1 dim
  - Has stereochemistry: 1 dim
- **edge_vector_features**: `[num_edges, 3]` (directional vectors)

#### Molecular-Level Features
- **mol_descriptors**: `[14]`
  - Molecular weight, LogP, TPSA, rotatable bonds, HBD, HBA, num atoms, num bonds, num rings, aromatic rings, heteroatom ratio, QED, Lipinski violations, fraction Csp3

#### Fingerprints (if requested)
- **MACCS keys**: `[166]` bits
- **Morgan fingerprint**: `[2048]` bits
- **RDKit fingerprint**: `[2048]` bits

### Protein Features (Macromolecules)

#### Node Features (Per Residue)
- **node_scalar_features**: Variable dimensions based on encoding
  - Residue type (one-hot): 20 dims for standard amino acids
  - Terminal flags: 2 dims (N-terminal, C-terminal)
  - Secondary structure: 8 dims (if available)
  - SASA (Solvent Accessible Surface Area): 1 dim
  - Phi/Psi angles: 2 dims
- **node_vector_features**: `[num_residues, 3, 3]`
  - Local coordinate frames (3x3 rotation matrices)
- **coordinates**: `[num_residues, 3]` (CA atom positions)

#### Edge Features (Residue Pairs)
- **edge_scalar_features**: `[num_edges, 4]`
  - CA-CA distance: 1 dim
  - SC-SC distance: 1 dim (side chain)
  - CA-SC distance: 1 dim
  - SC-CA distance: 1 dim
- **edge_vector_features**: `[num_edges, 3]`
  - 3D directional vectors between residues

## Molecule Features Extracted (Small Molecules)

### Physicochemical Descriptors
- Molecular weight, LogP, TPSA
- Number of rotatable bonds, flexibility
- Hydrogen bond donors/acceptors
- Number of atoms, bonds, rings
- Heteroatom ratio

### Drug-likeness Features
- Lipinski's Rule of Five violations
- QED (Quantitative Estimate of Drug-likeness)
- Fraction of sp3 carbons
- Number of heavy atoms

### Atom Composition
- Nitrogen, oxygen, sulfur, halogen, phosphorus ratios
- Universal atom type distributions

### Structural Features
- Ring systems count and sizes
- Average and maximum ring sizes

### Molecular Fingerprints
- MACCS keys (166 bits)
- Morgan fingerprints (2048 bits)
- RDKit fingerprints (2048 bits)
- Atom pair fingerprints
- Topological torsion fingerprints
- 2D pharmacophore fingerprints

## Protein Features Extracted (Macromolecules)

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

### Molecule Featurizer

```python
from featurizer import MoleculeFeaturizer

featurizer = MoleculeFeaturizer(add_hs=True)
features = featurizer.extract(mol_or_smiles)
```

**Parameters:**
- `mol_or_smiles`: RDKit mol object or SMILES string
- `add_hs`: Whether to add hydrogens (default: True)
- `compute_3d`: Whether to compute 3D conformers (default: False)

**Returns:**
Dictionary containing:
- `descriptors`: Tensor `[14]` of physicochemical descriptors
- `fingerprints`: Dictionary of fingerprints:
  - `maccs`: MACCS keys `[166]`
  - `morgan`: Morgan fingerprint `[2048]`
  - `rdkit`: RDKit fingerprint `[2048]`
- `graph`: Node/edge format (if requested):
  - `node`: Dictionary with scalar `[num_atoms, 26]`, vector `[num_atoms, 1]`, coord `[num_atoms, 3]`
  - `edge`: Dictionary with scalar `[num_edges, 7]`, vector `[num_edges, 3]`

### Protein Featurizer

```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer(standardize=True, keep_hydrogens=False)
features = featurizer.extract(pdb_file)
```

**Parameters:**
- `standardize`: Whether to standardize PDB (default: True)
- `keep_hydrogens`: Whether to keep hydrogens (default: False)

**Returns:**
Dictionary containing:
- `node`: Node features dictionary
  - `coord`: `[num_residues, 3]` CA positions
  - `node_scalar_features`: Variable dimensions based on encoding
  - `node_vector_features`: `[num_residues, 3, 3]` local frames
- `edge`: Edge features dictionary
  - `edges`: `(src_indices, dst_indices)` connectivity
  - `edge_scalar_features`: `[num_edges, 4]` distances
  - `edge_vector_features`: `[num_edges, 3]` directional vectors

**Methods:**
- `extract(pdb_file, save_to=None)`: Extract features from a single PDB file
- `extract_batch(pdb_files, output_dir=None, skip_existing=True, verbose=True)`: Process multiple files
- `from_clean_pdb(pdb_file)`: Extract from pre-cleaned PDB (class method)
- `standardize_only(input_pdb, output_pdb, keep_hydrogens=False)`: Only standardize PDB (static method)

## Advanced Usage

### Accessing Feature Dimensions

```python
from featurizer import MoleculeFeaturizer, MoleculeFeatureExtractor
from rdkit import Chem

# Extract molecule features in node/edge format
extractor = MoleculeFeatureExtractor()
node, edge = extractor.get_features("CCO")

print(f"Node scalar features shape: {node['node_scalar_features'][0].shape}")  # [num_atoms, 26]
print(f"Node vector features shape: {node['node_vector_features'][0].shape}")  # [num_atoms, 1]
print(f"Coordinates shape: {node['coord'].shape}")                            # [num_atoms, 3]
print(f"Molecular descriptors shape: {node['mol_descriptors'].shape}")        # [14]

print(f"Edge scalar features shape: {edge['edge_scalar_features'][0].shape}") # [num_edges, 7]
print(f"Edge vector features shape: {edge['edge_vector_features'][0].shape}") # [num_edges, 3]
```

### Custom Molecule Feature Extraction

```python
from featurizer import MoleculeFeaturizer, MoleculeFeatureExtractor
from rdkit import Chem

# Using high-level API
featurizer = MoleculeFeaturizer()
features = featurizer.extract("CCO")

# Using low-level API for custom control
mol = Chem.MolFromSmiles("CCO")
extractor = MoleculeFeatureExtractor()

# Get specific feature types
phys_features = extractor.get_physicochemical_features(mol)
fingerprints = extractor.get_fingerprints(mol)

# Get node/edge format for graph neural networks
node, edge = extractor.get_features(mol)
```

### Batch Processing

```python
from featurizer import ProteinFeaturizer, MoleculeFeaturizer

# Process multiple proteins
protein_featurizer = ProteinFeaturizer()
pdb_files = ["1abc.pdb", "2def.pdb", "3ghi.pdb"]
results = protein_featurizer.extract_batch(
    pdb_files,
    output_dir="processed_features/",
    skip_existing=True,
    verbose=True
)

# Process multiple molecules
molecule_featurizer = MoleculeFeaturizer()
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
results = molecule_featurizer.extract_batch(
    smiles_list,
    output_dir="molecule_features/",
    skip_existing=True,
    verbose=True
)
pdb_files = ["1abc.pdb", "2def.pdb", "3ghi.pdb"]
results = featurizer.extract_batch(
    pdb_files,
    output_dir="processed_features/",
    skip_existing=True,
    verbose=True
)
```

## Requirements

- Python ≥ 3.7
- RDKit ≥ 2020.09
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- Pandas ≥ 1.3.0
- FreeSASA ≥ 2.1.0 (for protein features)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{featurizer,
  title = {Featurizer: A Python package for molecule and protein structure feature extraction},
  author = {Jaemin Sim},
  year = {2025},
  url = {https://github.com/eightmm/featurizer}
}
```
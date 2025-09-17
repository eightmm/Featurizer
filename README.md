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
from featurizer.molecule_featurizer import create_molecule_features
from rdkit import Chem

# From SMILES string
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
features = create_molecule_features(smiles)

# From RDKit mol object
mol = Chem.MolFromSmiles(smiles)
features = create_molecule_features(mol)

# Access different feature types
descriptors = features['descriptor']  # Physicochemical descriptors
morgan_fp = features['morgan']  # Morgan fingerprint
maccs_fp = features['maccs']  # MACCS keys
```

### Protein Features (Macromolecules)

```python
from featurizer.protein_featurizer import Featurizer

# Initialize featurizer
featurizer = Featurizer()

# Extract features from a PDB file
features = featurizer.extract("protein.pdb")

# Access node and edge features
node_features = features['node']
edge_features = features['edge']

# Save features to file
features = featurizer.extract("protein.pdb", save_to="features.pt")
```

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
create_molecule_features(mol_or_smiles, add_hs=True)
```

**Parameters:**
- `mol_or_smiles`: RDKit mol object or SMILES string
- `add_hs`: Whether to add hydrogens (default: True)

**Returns:**
Dictionary containing:
- `descriptor`: Tensor of physicochemical descriptors
- `maccs`: MACCS fingerprint
- `morgan`: Morgan fingerprint
- `morgan_count`: Morgan count fingerprint
- `feature_morgan`: Feature Morgan fingerprint
- `rdkit`: RDKit fingerprint
- `atom_pair`: Atom pair fingerprint
- `topological_torsion`: Topological torsion fingerprint
- `pharmacophore2d`: 2D pharmacophore fingerprint

### Protein Featurizer

```python
Featurizer(standardize=True, keep_hydrogens=False)
```

**Methods:**
- `extract(pdb_file, save_to=None)`: Extract features from a single PDB file
- `extract_batch(pdb_files, output_dir=None, skip_existing=True, verbose=True)`: Process multiple files
- `from_clean_pdb(pdb_file)`: Extract from pre-cleaned PDB (class method)
- `standardize_only(input_pdb, output_pdb, keep_hydrogens=False)`: Only standardize PDB (static method)

## Advanced Usage

### Custom Molecule Feature Extraction

```python
from featurizer.molecule_featurizer import MoleculeFeatureExtractor
from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
extractor = MoleculeFeatureExtractor()

# Get specific feature types
phys_features = extractor.get_physicochemical_features(mol)
drug_features = extractor.get_druglike_features(mol)
struct_features = extractor.get_structural_features(mol)
fingerprints = extractor.get_fingerprints(mol)
```

### Batch Processing

```python
from featurizer.protein_featurizer import Featurizer

# Process multiple proteins
featurizer = Featurizer()
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
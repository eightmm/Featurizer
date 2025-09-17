# Featurizer Package Refactoring Summary

## Overview
The package has been refactored from `protein-featurizer` to `featurizer`, a comprehensive toolkit for extracting features from both molecular and protein structures.

## Major Changes

### 1. Package Renaming
- **Old**: `protein-featurizer`
- **New**: `featurizer`
- **Version**: Updated from 0.1.0 to 0.2.0

### 2. Molecular Feature Module Cleanup

#### Removed CYP3A4-Specific Features
- Removed `CYP3A4_INHIBITOR_SMARTS` dictionary with task-specific patterns
- Removed `CYP3A4_SUBSTRATE_SMARTS` dictionary with task-specific patterns
- Removed `get_cyp3a4_features()` method

#### Added Universal Features
- Added `get_atom_composition_features()` method for universal atom ratios:
  - Nitrogen, oxygen, sulfur, halogen, and phosphorus ratios
  - These are generally applicable to all molecular analyses

#### API Improvements
- `extract_all_features()` now accepts RDKit mol objects directly
- Added `add_hs` parameter to control hydrogen addition
- `create_molecular_features()` supports both SMILES strings and mol objects

### 3. Package Structure

```
featurizer/
├── __init__.py                 # Main package with convenience functions
├── molecule_featurizer/
│   ├── __init__.py
│   ├── molecular_feature.py    # Cleaned universal features
│   └── molecular_graph.py      # Graph representation
└── protein_featurizer/
    ├── __init__.py
    ├── main.py
    ├── pdb_standardizer.py
    └── residue_featurizer.py
```

### 4. Features Retained

#### Molecular Features (Universal)
- **Physicochemical**: MW, LogP, TPSA, rotatable bonds, HBD/HBA
- **Drug-likeness**: Lipinski violations, QED score, Fsp3
- **Structural**: Ring systems, ring sizes
- **Atom Composition**: Universal atom type ratios
- **Fingerprints**: MACCS, Morgan, RDKit, Atom Pair, Topological Torsion, Pharmacophore

#### Protein Features
- All existing protein features remain unchanged
- PDB standardization and residue featurization

### 5. Updated Files

| File | Changes |
|------|---------|
| `molecular_feature.py` | Removed CYP3A4 patterns, added universal features, accepts mol objects |
| `setup.py` | Renamed package, updated metadata and entry points |
| `README.md` | Complete rewrite for new package structure |
| `requirements.txt` | Added rdkit dependency |
| `__init__.py` files | Created/updated for new package structure |
| `examples/` | Updated all examples for new import structure |

### 6. New API Examples

```python
# Molecular features from SMILES
from featurizer import create_molecular_features
features = create_molecular_features("CCO")

# Molecular features from RDKit mol
from rdkit import Chem
mol = Chem.MolFromSmiles("CCO")
features = create_molecular_features(mol, add_hs=False)

# Protein features
from featurizer import ProteinFeaturizer
featurizer = ProteinFeaturizer()
features = featurizer.extract("protein.pdb")
```

### 7. Benefits of Refactoring

1. **Universal Applicability**: Features are now task-agnostic and universally applicable
2. **Flexible Input**: Supports both SMILES strings and RDKit mol objects
3. **Unified Package**: Single package for both molecular and protein features
4. **Clean API**: Simplified and more intuitive function signatures
5. **Maintainability**: Cleaner code structure without task-specific patterns

## Migration Guide

### For Existing Users

1. **Import Changes**:
   ```python
   # Old
   from protein_featurizer import Featurizer

   # New
   from featurizer import ProteinFeaturizer
   ```

2. **Molecular Features**:
   ```python
   # New capability
   from featurizer import create_molecular_features
   features = create_molecular_features(smiles_or_mol)
   ```

3. **Removed Features**:
   - CYP3A4-specific pattern matching is no longer available
   - Use general atom composition features instead

## Installation

```bash
# From GitHub
pip install git+https://github.com/eightmm/featurizer.git

# For development
git clone https://github.com/eightmm/featurizer.git
cd featurizer
pip install -e .
```

## Dependencies

- torch >= 1.9.0
- numpy >= 1.19.0
- pandas >= 1.3.0
- rdkit >= 2020.09.1
- freesasa >= 2.1.0 (for protein features)

## Testing

A test script is provided to verify the refactoring:

```bash
python test_refactoring.py
```

This will check:
- Module imports
- Removal of CYP3A4-specific code
- New API functionality
- RDKit mol object support
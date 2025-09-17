# Molecule Features Documentation

## Overview
The molecule featurizer extracts comprehensive molecular descriptors and fingerprints for machine learning applications.

## 1. Molecular Descriptors (`get_feature()`)

Extracts 40 normalized molecular descriptors covering physicochemical, topological, and structural properties.

### Usage
```python
from featurizer import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()
features = featurizer.get_feature("CCO")
descriptors = features['descriptor']  # torch.Tensor [40]
```

### Descriptor Categories

#### Basic Physicochemical (12 descriptors)
- `mw`: Molecular weight (/1000.0)
- `logp`: Octanol-water partition coefficient ((+5)/10.0)
- `tpsa`: Topological polar surface area (/200.0)
- `n_rotatable_bonds`: Number of rotatable bonds (/20.0)
- `flexibility`: Rotatable bonds ratio (0-1)
- `hbd`: Hydrogen bond donors (/10.0)
- `hba`: Hydrogen bond acceptors (/15.0)
- `n_atoms`: Total atoms (/100.0)
- `n_bonds`: Total bonds (/120.0)
- `n_rings`: Number of rings (/10.0)
- `n_aromatic_rings`: Aromatic rings (/8.0)
- `heteroatom_ratio`: Heteroatoms ratio (0-1)

#### Topological Indices (9 descriptors)
- `balaban_j`: Balaban's J index (/5.0)
- `bertz_ct`: Bertz complexity (/2000.0)
- `chi0`, `chi1`: Connectivity indices
- `chi0n`: Normalized connectivity
- `hall_kier_alpha`: Hall-Kier alpha (/5.0)
- `kappa1`, `kappa2`, `kappa3`: Kappa shape indices

#### Electronic Properties (4 descriptors)
- `mol_mr`: Molar refractivity (/200.0)
- `labute_asa`: Accessible surface area (/500.0)
- `num_radical_electrons`: Radical electrons (/5.0)
- `num_valence_electrons`: Valence electrons (/500.0)

#### Structural Features (10 descriptors)
- Ring types: saturated, aliphatic, heterocycles
- `num_heteroatoms`: Total heteroatoms (/30.0)
- `formal_charge`: Sum of formal charges ((+5)/10.0)
- `n_ring_systems`: Ring systems (/8.0)
- `max_ring_size`: Maximum ring size (/12.0)
- `avg_ring_size`: Average ring size (/8.0)

#### Drug-likeness (5 descriptors)
- `lipinski_violations`: Rule of 5 violations (0-1)
- `passes_lipinski`: Binary flag (0 or 1)
- `qed`: Quantitative Estimate of Drug-likeness (0-1)
- `num_heavy_atoms`: Heavy atoms (/50.0)
- `frac_csp3`: Fraction of sp3 carbons (0-1)

## 2. Molecular Fingerprints

Nine types of molecular fingerprints for similarity searching and machine learning.

### Available Fingerprints

#### Morgan Fingerprint
```python
features['morgan']  # torch.Tensor [2048]
```
Circular fingerprint capturing local chemical environments (radius=2).

#### MACCS Keys
```python
features['maccs']  # torch.Tensor [167]
```
166 predefined structural keys + 1 padding for consistency.

#### RDKit Fingerprint
```python
features['rdkit']  # torch.Tensor [2048]
```
Path-based fingerprint encoding molecular substructures.

#### Atom Pair Fingerprint
```python
features['atompair']  # torch.Tensor [2048]
```
Encodes pairs of atoms and their topological distances.

#### Topological Torsion
```python
features['torsion']  # torch.Tensor [2048]
```
Four-atom linear paths through the molecule.

#### Avalon Fingerprint
```python
features['avalon']  # torch.Tensor [1024]
```
Feature-class based fingerprint (requires AvalonTools).

#### Pattern Fingerprint
```python
features['pattern']  # torch.Tensor [2048]
```
Substructure pattern-based encoding.

#### Extended Connectivity
```python
features['ecfp4']  # torch.Tensor [2048]
```
Circular fingerprint with radius=2 (similar to ECFP4).

#### Functional Connectivity
```python
features['fcfp4']  # torch.Tensor [2048]
```
Feature-based circular fingerprint.

### Fingerprint Usage Example
```python
from featurizer import MoleculeFeaturizer
import torch

featurizer = MoleculeFeaturizer()

# Get all fingerprints
features = featurizer.get_feature("c1ccccc1")
morgan = features['morgan']
maccs = features['maccs']

# Similarity calculation
def tanimoto_similarity(fp1, fp2):
    intersection = torch.sum(torch.min(fp1, fp2))
    union = torch.sum(torch.max(fp1, fp2))
    return intersection / union

mol1_features = featurizer.get_feature("CCO")
mol2_features = featurizer.get_feature("CCN")
similarity = tanimoto_similarity(mol1_features['morgan'], mol2_features['morgan'])
```

## 3. Combined Features

The `get_feature()` method returns all features in a single dictionary:

```python
features = featurizer.get_feature(mol)
# Returns:
{
    'descriptor': torch.Tensor,  # [40] molecular descriptors
    'morgan': torch.Tensor,      # [2048] Morgan fingerprint
    'maccs': torch.Tensor,       # [167] MACCS keys
    'rdkit': torch.Tensor,       # [2048] RDKit fingerprint
    'atompair': torch.Tensor,    # [2048] Atom pair
    'torsion': torch.Tensor,     # [2048] Topological torsion
    'avalon': torch.Tensor,      # [1024] Avalon
    'pattern': torch.Tensor,     # [2048] Pattern
    'ecfp4': torch.Tensor,       # [2048] ECFP4
    'fcfp4': torch.Tensor        # [2048] FCFP4
}
```

## Input Formats

Accepts both RDKit mol objects and SMILES strings:

```python
from rdkit import Chem

# From SMILES
features = featurizer.get_feature("CCO")

# From RDKit mol
mol = Chem.MolFromSmiles("CCO")
features = featurizer.get_feature(mol)

# From SDF file
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    features = featurizer.get_feature(mol)
```

## Performance
- **Descriptors only**: ~1ms per molecule
- **All fingerprints**: ~5ms per molecule
- **Complete features**: ~6ms per molecule

## Applications
- **Virtual Screening**: Fingerprints for similarity searching
- **QSAR/QSPR**: Descriptors for property prediction
- **Machine Learning**: Combined features for classification/regression
- **Clustering**: Chemical space exploration
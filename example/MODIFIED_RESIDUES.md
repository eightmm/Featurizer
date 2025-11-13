# Modified Amino Acid Support

The PDB standardizer now supports automatic conversion of 40+ modified amino acids to their standard counterparts.

## Supported Modifications

### Selenomethionine (Common in X-ray Crystallography)
- **MSE** → MET (Selenomethionine, Se replaces S)
- **FME** → MET (N-formylmethionine)

### Post-Translational Modifications (PTMs)

#### Phosphorylation
- **SEP** → SER (Phosphoserine)
- **TPO** → THR (Phosphothreonine)
- **PTR** → TYR (Phosphotyrosine)

#### Methylation
- **MLY** → LYS (N-dimethyllysine)
- **M3L** → LYS (N-trimethyllysine)
- **MEN** → ASN (N-methylasparagine)

#### Acetylation
- **ALY** → LYS (N-acetyllysine)

#### Oxidation
- **CSO** → CYS (S-hydroxycysteine)
- **CSS** → CYS (S-mercaptocysteine)
- **CME** → CYS (S-methylcysteine)
- **OCS** → CYS (Cysteinesulfonic acid)

#### Hydroxylation
- **HYP** → PRO (Hydroxyproline, common in collagen)

### Protonation States

#### Histidine (neutral and charged forms)
- **HID** → HIS (δ-protonated, neutral)
- **HIE** → HIS (ε-protonated, neutral)
- **HIP** → HIS (doubly protonated, positive)
- **HSD** → HIS (CHARMM δ-protonated)
- **HSE** → HIS (CHARMM ε-protonated)
- **HSP** → HIS (CHARMM doubly protonated)
- **HIN** → HIS (alternative neutral)

#### Cysteine (disulfide bonding states)
- **CYX** → CYS (disulfide-bonded)
- **CYM** → CYS (deprotonated thiolate)
- **CYN** → CYS (alternative deprotonated)

#### Acidic Residues (protonation states)
- **ASH** → ASP (protonated aspartic acid)
- **ASPP** → ASP (alternative protonated)
- **GLH** → GLU (protonated glutamic acid)
- **GLUP** → GLU (alternative protonated)
- **GLUH** → GLU (alternative protonated)

#### Lysine (protonation states)
- **LYN** → LYS (deprotonated, neutral)
- **LYSN** → LYS (alternative deprotonated)

#### Other Protonation States
- **ARN** → ARG (deprotonated arginine)
- **TYM** → TYR (deprotonated tyrosinate)
- **TYN** → TYR (alternative deprotonated)

## Usage

All conversions happen automatically during standardization:

```python
from featurizer import PDBStandardizer

# Automatically converts modified residues
standardizer = PDBStandardizer()
standardizer.standardize("protein_with_mse.pdb", "protein_clean.pdb")

# MSE, SEP, TPO, etc. are automatically converted to MET, SER, THR
```

## Example

```python
from featurizer import ProteinFeaturizer

# Input PDB has MSE (selenomethionine) residues
featurizer = ProteinFeaturizer("protein_with_mse.pdb", standardize=True)

# After standardization, MSE is converted to MET
sequences = featurizer.get_sequence_by_chain()
# Sequences will show 'M' for MET (not 'X' for unknown)
```

## Why This Matters

1. **X-ray Crystallography**: MSE is commonly used for phasing
2. **PTMs**: Phosphorylation, methylation are biologically important
3. **MD Simulations**: Force fields use different protonation states
4. **Compatibility**: Many tools only recognize 20 standard amino acids

## Implementation

All mappings preserve heavy atom structure - only atom types change (e.g., Se→S for MSE→MET). This ensures coordinates remain valid while making the structure compatible with standard analysis tools.

## Testing

Run the PTM mapping test:

```bash
cd example
python test_ptm_mappings.py
```

Expected output:
```
✓  MSE    → MET    : FOUND
✓  SEP    → SER    : FOUND
✓  TPO    → THR    : FOUND
✓  PTR    → TYR    : FOUND
✓  HYP    → PRO    : FOUND
✓  MLY    → LYS    : FOUND
✓  CSO    → CYS    : FOUND

✓ ALL PTM MAPPINGS WORKING CORRECTLY!
```

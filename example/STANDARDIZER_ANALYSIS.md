# PDB Standardizer Analysis Report

## Summary

The PDB standardizer works correctly for most common cases but has some limitations to be aware of.

## Test Results

### ✓ Working Correctly

1. **Hydrogen Removal** - Removes H atoms when requested
2. **Water Removal** - Removes HOH/WAT molecules
3. **DNA/RNA Removal** - Removes nucleic acid residues
4. **Residue Name Normalization** - Converts HID/HIE/HIP → HIS, CYX → CYS, etc.
5. **Atom Reordering** - Sorts atoms by standard order (N, CA, C, O, CB, ...)
6. **Chain Grouping** - Groups residues by chain correctly
7. **Alternative Locations** - Keeps first occurrence (implicit A selection)

### ⚠️ Limitations Found

#### 1. Insertion Codes are REMOVED

**Issue**: Residues with insertion codes (e.g., 100A, 100B) are renumbered sequentially.

```
Input:  Chain A: 10, 10A, 10B, 11
Output: Chain A: 1, 2, 3, 4
```

**Impact**:
- Residue numbering in standardized PDB does NOT match original PDB
- If you need to map back to original residue numbers, you'll lose this information
- Insertion codes provide important structural information (often used for loops)

**Current Behavior**: Sequential renumbering starting from 1 per chain

**Recommendation**:
- Document this behavior clearly
- Consider preserving insertion codes in future versions
- Or provide mapping file: old_resnum → new_resnum

#### 2. MSE (Selenomethionine) Not Mapped

**Issue**: MSE is a common modified residue (MET with Se instead of S) but not in the mapping.

```
Input:  MSE (selenomethionine)
Output: MSE (unchanged, not converted to MET)
```

**Impact**:
- MSE residues are treated as non-standard residues
- May cause issues with tools expecting only 20 standard amino acids
- MSE is very common in X-ray structures (used for phasing)

**Recommendation**: Add to `RESIDUE_NAME_MAPPING`:
```python
'MSE': 'MET',  # Selenomethionine → Methionine
```

#### 3. Alternative Location Handling is Implicit

**Issue**: When multiple altlocs exist (A, B), the first one encountered is kept.

```
Input:  CA A (occupancy 0.5), CA B (occupancy 0.5)
Output: CA A (first one found)
```

**Current Behavior**: Works, but not optimal

**Recommendation**:
- Pick highest occupancy altloc
- Or explicitly pick 'A' if occupancies are equal
- Document the selection strategy

## Missing from Mapping

Common modified amino acids NOT in current mapping:

```python
# Selenomethionine (very common in crystallography)
'MSE': 'MET',

# Pyrrolysine (22nd amino acid, rare but valid)
'PYL': 'LYS',  # or keep as PYL?

# Other common modifications
'SEP': 'SER',  # Phosphoserine
'TPO': 'THR',  # Phosphothreonine
'PTR': 'TYR',  # Phosphotyrosine
'HYP': 'PRO',  # Hydroxyproline
'MLY': 'LYS',  # N-dimethyllysine
'M3L': 'LYS',  # N-trimethyllysine
```

## Code Quality

### Good Practices
- ✓ Clear separation of concerns (parse, process, write)
- ✓ Comprehensive residue name mapping for protonation states
- ✓ Proper PDB format adherence (column positions)
- ✓ Chain-based processing
- ✓ Handles HETATM separately

### Potential Improvements

1. **Altloc Selection**
```python
# Current: implicit first-found selection
# Better: explicit occupancy-based selection
if altloc and atom_name in residue_atoms:
    # Compare occupancies, keep highest
    pass
```

2. **Insertion Code Preservation**
```python
# Option 1: Preserve insertion codes
res_num_with_ins = f"{res_counter}{insertion_code}" if insertion_code else f"{res_counter}"

# Option 2: Provide mapping output
mapping[f"{chain}:{original_resnum}"] = f"{chain}:{new_resnum}"
```

3. **Add MSE Mapping**
```python
RESIDUE_NAME_MAPPING = {
    # ... existing mappings ...
    'MSE': 'MET',  # Selenomethionine
}
```

## Usage Recommendations

### When Standardizer Works Well
- ✓ Removing hydrogens for analysis
- ✓ Cleaning up protonation state variations
- ✓ Removing water/nucleic acids
- ✓ Standardizing atom order
- ✓ Sequential renumbering needed

### When to Be Careful
- ⚠️ If you need to preserve original residue numbering
- ⚠️ If insertion codes are important for your analysis
- ⚠️ If you have MSE residues (add to mapping first)
- ⚠️ If you need specific altloc handling

### Workarounds

**For Insertion Codes**:
```python
# Store original residue numbers before standardization
from featurizer import ProteinFeaturizer

# Option 1: Don't standardize
featurizer = ProteinFeaturizer("protein.pdb", standardize=False)

# Option 2: Create your own mapping
original_residues = parse_original_pdb("protein.pdb")
standardized_residues = parse_standardized_pdb("protein_clean.pdb")
mapping = create_residue_mapping(original_residues, standardized_residues)
```

**For MSE**:
```python
# Temporarily add MSE to mapping
from featurizer.protein_featurizer.pdb_standardizer import RESIDUE_NAME_MAPPING
RESIDUE_NAME_MAPPING['MSE'] = 'MET'

# Then standardize
from featurizer import PDBStandardizer
standardizer = PDBStandardizer()
standardizer.standardize("input.pdb", "output.pdb")
```

## Test Coverage

Tested scenarios:
- ✓ Insertion codes (10, 10A, 10B)
- ✓ Alternative locations (A/B altlocs)
- ✓ Modified amino acids (MSE, HID)
- ✓ Residue name formatting
- ✓ Hydrogen removal
- ✓ Water removal
- ✓ Chain grouping

## Conclusion

The PDB standardizer is **functional and useful** for most common cases:
- Cleaning up PDB files for analysis
- Removing unwanted molecules (water, nucleic acids, hydrogens)
- Normalizing residue names (protonation states)
- Standardizing atom order

**Key Limitation**: Insertion codes are removed and residues are renumbered sequentially. This is fine for many applications but important to be aware of.

**Recommended Improvements**:
1. Add MSE → MET mapping (very common)
2. Document insertion code removal behavior
3. Consider preserving insertion codes or providing mapping
4. Improve altloc selection (occupancy-based)

The standardizer is production-ready with these limitations documented.

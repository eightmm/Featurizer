# Chain-based Sequence Extraction Feature

## Summary

Added new method to extract protein sequences in one-letter amino acid code, separated by chain.

## New Method

### ResidueFeaturizer & ProteinFeaturizer

**`get_sequence_by_chain() -> Dict[str, str]`**
- Returns sequences separated by chain ID
- Example: `{'A': 'ACDEF...', 'B': 'GHIKL...'}`

```python
from featurizer import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Get sequences by chain
chain_seqs = featurizer.get_sequence_by_chain()
for chain_id, seq in chain_seqs.items():
    print(f"Chain {chain_id}: {seq}")

# If you need full sequence, concatenate:
full_seq = ''.join(chain_seqs.values())
```

## Implementation Details

### Conversion Logic

- Uses existing `AMINO_ACID_3TO1` mapping dictionary
- Converts residue type integers back to 3-letter codes
- Then converts 3-letter codes to 1-letter codes
- Unknown residues are mapped to 'X'

### Chain Handling

- Sequences are grouped by chain ID from PDB file
- Chain IDs are preserved as keys in the dictionary
- Residues are sorted by chain and residue number

## Files Modified

1. **`featurizer/protein_featurizer/residue_featurizer.py`**
   - Added `get_sequence_by_chain()` method

2. **`featurizer/protein_featurizer/protein_featurizer.py`**
   - Exposed `get_sequence_by_chain()` method in ProteinFeaturizer class

3. **`README.md`**
   - Added sequence extraction examples
   - Updated quick start section
   - Added advanced usage example

## Example Usage

See `claudedocs/sequence_by_chain_example.py` for a complete working example.

```bash
python claudedocs/sequence_by_chain_example.py protein.pdb
```

## Testing

To test the functionality:

```python
from featurizer import ProteinFeaturizer

# Test with your PDB file
featurizer = ProteinFeaturizer("your_protein.pdb")

# Test chain-based sequences
chain_seqs = featurizer.get_sequence_by_chain()
assert isinstance(chain_seqs, dict)
assert all(isinstance(s, str) for s in chain_seqs.values())

# Test that we can get full sequence if needed
full_seq = ''.join(chain_seqs.values())
assert len(full_seq) > 0

print("✓ All tests passed!")
```

## Backward Compatibility

- All existing functionality is preserved
- No breaking changes to existing APIs
- New methods are additive only

## Future Enhancements

Potential future improvements:
- Add FASTA format export
- Add sequence metadata (chain boundaries, modified residues)
- Support for custom residue mappings
- Integration with sequence alignment tools

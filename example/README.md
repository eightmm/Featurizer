# Featurizer Examples

This directory contains example files and scripts for testing the Featurizer package.

## Files

- **10gs_protein.pdb**: Example protein structure (Glutathione S-transferase, 416 residues, 2 chains)
- **10gs_ligand.sdf**: Example small molecule ligand
- **test_featurizer.py**: Comprehensive test script demonstrating all features

## Running the Tests

```bash
# Make sure you're in the example directory
cd example

# Run the comprehensive test
python test_featurizer.py
```

## What the Test Covers

### Protein Featurizer Tests

1. **Sequence Extraction** - Extract amino acid sequences by chain
2. **Residue-Level Features** - Node and edge features at residue level
3. **Atom-Level Features** - Node and edge features at atom level
4. **SASA Features** - Solvent accessible surface area calculations
5. **Contact Maps** - Residue-residue interaction networks
6. **Geometric Features** - Dihedrals, curvature, torsion angles

### Molecule Featurizer Tests

1. **Basic Properties** - SMILES, formula, atom/bond counts
2. **Descriptors** - 40 normalized molecular descriptors
3. **Fingerprints** - 7 types (Morgan, MACCS, RDKit, etc.)
4. **Graph Representation** - Node/edge features with adjacency matrix
5. **Hydrogen Handling** - Comparison with/without hydrogens
6. **Custom SMARTS** - User-defined structural pattern matching

## Expected Output

When you run the test script, you should see output like:

```
======================================================================
  FEATURIZER PACKAGE - COMPREHENSIVE TESTING
======================================================================

📂 Working directory: /path/to/example

======================================================================
  PROTEIN FEATURIZER TESTS
======================================================================

✓ Protein loaded successfully

  Chain A: 208 residues
  Chain B: 208 residues
  Total residues: 416

... [detailed feature information] ...

✓ All protein tests completed successfully!

======================================================================
  MOLECULE FEATURIZER TESTS
======================================================================

✓ Molecule loaded successfully

... [detailed feature information] ...

✓ All molecule tests completed successfully!

======================================================================
  ✓ ALL TESTS COMPLETED SUCCESSFULLY!
======================================================================
```

## Using Your Own Files

You can modify the test script to use your own PDB and SDF files:

```python
# In test_featurizer.py, change the file paths:
pdb_file = "your_protein.pdb"
sdf_file = "your_ligand.sdf"
```

Or create a new script based on the examples:

```python
from featurizer import ProteinFeaturizer, MoleculeFeaturizer

# For proteins
protein = ProteinFeaturizer("your_protein.pdb")
sequences = protein.get_sequence_by_chain()
res_node, res_edge = protein.get_residue_features()

# For molecules
from rdkit import Chem
mol = Chem.SDMolSupplier("your_ligand.sdf")[0]
molecule = MoleculeFeaturizer(mol)
features = molecule.get_feature()
node, edge, adj = molecule.get_graph()
```

## Troubleshooting

### Missing Dependencies

If you get import errors, make sure all dependencies are installed:

```bash
pip install rdkit-pypi torch numpy pandas freesasa
```

### File Not Found

Make sure you're running the script from the `example` directory:

```bash
cd /path/to/Featurizer/example
python test_featurizer.py
```

### FreeSASA Warnings

You may see warnings about unknown atoms from FreeSASA. These are normal and can be ignored - the package handles them internally.

## More Information

- See the main [README.md](../README.md) for API documentation
- See [docs/](../docs/) for detailed feature descriptions
- See [claudedocs/](../claudedocs/) for additional examples

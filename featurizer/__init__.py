"""
Featurizer Package

A comprehensive Python package for extracting features from molecular and protein structures
for machine learning applications.
"""

# Import molecule featurizer components
from .molecule_featurizer import (
    MoleculeFeaturizer,
    MoleculeFeatureExtractor,
    create_molecule_features
)

# Import protein featurizer components
from .protein_featurizer import (
    ProteinFeaturizer,
    PDBStandardizer,
    ResidueFeaturizer,
    standardize_pdb,
    process_pdb,
    batch_process
)

__version__ = "0.2.0"
__author__ = "Jaemin Sim"
__email__ = "your.email@example.com"

__all__ = [
    # Molecule features
    "MoleculeFeaturizer",
    "MoleculeFeatureExtractor",
    "create_molecule_features",
    # Protein features
    "ProteinFeaturizer",
    "PDBStandardizer",
    "ResidueFeaturizer",
    "standardize_pdb",
    "process_pdb",
    "batch_process",
]

# Convenience functions for quick access
def extract_molecule_features(mol_or_smiles, add_hs=True):
    """
    Convenience function to extract molecule features.

    Args:
        mol_or_smiles: RDKit mol object or SMILES string
        add_hs: Whether to add hydrogens (default: True)

    Returns:
        Dictionary containing molecule features
    """
    return create_molecule_features(mol_or_smiles, add_hs)

def extract_protein_features(pdb_file, standardize=True, save_to=None):
    """
    Convenience function to extract protein features.

    Args:
        pdb_file: Path to PDB file
        standardize: Whether to standardize PDB first (default: True)
        save_to: Optional path to save features

    Returns:
        Dictionary containing protein features
    """
    featurizer = ProteinFeaturizer(standardize=standardize)
    return featurizer.extract(pdb_file, save_to=save_to)
"""
Molecule Featurizer Module

A comprehensive toolkit for extracting molecular features from SMILES and RDKit mol objects.
"""

from .molecule_feature import (
    MoleculeFeatureExtractor,
    create_molecule_features
)

from .molecule_graph import (
    MoleculeGraphBuilder,
    create_molecule_graph
)

__version__ = "0.2.0"
__author__ = "Jaemin Sim"

__all__ = [
    "MoleculeFeatureExtractor",
    "create_molecule_features",
    "MoleculeGraphBuilder",
    "create_molecule_graph",
]
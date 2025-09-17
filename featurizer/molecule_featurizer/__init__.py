"""
Molecule Featurizer Module

A comprehensive toolkit for extracting molecular features from SMILES and RDKit mol objects.
"""

from .molecular_feature import (
    MolecularFeatureExtractor,
    create_molecular_features
)

from .molecular_graph import (
    MolecularGraph,
    create_molecular_graph,
    smiles_to_graph
)

__version__ = "0.2.0"
__author__ = "Jaemin Sim"

__all__ = [
    "MolecularFeatureExtractor",
    "create_molecular_features",
    "MolecularGraph",
    "create_molecular_graph",
    "smiles_to_graph",
]
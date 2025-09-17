"""
Efficient Molecule Featurizer with optional caching.

This module provides an enhanced API for molecule feature extraction
with optional caching for repeated feature access.
"""

from typing import Optional, Union, Dict, Any, Tuple
import torch
from rdkit import Chem
from .molecule_feature import MoleculeFeaturizer as CoreFeaturizer


class MoleculeFeaturizer:
    """
    Enhanced molecule featurizer with efficient caching.

    Can be used in two modes:
    1. Instance mode: Initialize with a molecule, then call get methods (efficient for repeated access)
    2. Static mode: Call methods with molecule as parameter (backward compatible)

    Examples:
        >>> # Instance mode (efficient for multiple features from same molecule)
        >>> featurizer = MoleculeFeaturizer("CC(=O)Oc1ccccc1C(=O)O")
        >>> features = featurizer.get_feature()
        >>> graph = featurizer.get_graph()

        >>> # Static mode (backward compatible, good for one-off extraction)
        >>> featurizer = MoleculeFeaturizer()
        >>> features = featurizer.get_feature("CC(=O)Oc1ccccc1C(=O)O")
    """

    def __init__(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
                 add_hs: bool = True):
        """
        Initialize featurizer, optionally with a molecule for efficient access.

        Args:
            mol_or_smiles: Optional molecule (RDKit mol or SMILES) to cache
            add_hs: Whether to add hydrogens to molecules
        """
        self._core = CoreFeaturizer()
        self.add_hs = add_hs
        self._mol = None
        self._cache = {}

        # If molecule provided, prepare and cache it
        if mol_or_smiles is not None:
            self._mol = self._core._prepare_mol(mol_or_smiles, add_hs)
            self._parse_molecule()

    def _parse_molecule(self):
        """Parse and cache basic molecular data."""
        if self._mol is None:
            return

        # Cache basic info
        self._cache['num_atoms'] = self._mol.GetNumAtoms()
        self._cache['num_bonds'] = self._mol.GetNumBonds()

        # Pre-calculate 3D status
        self._cache['has_3d'] = self._mol.GetNumConformers() > 0

    def _get_mol(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> Chem.Mol:
        """
        Get molecule to process, using cached if available.

        Args:
            mol_or_smiles: Optional molecule override

        Returns:
            Prepared molecule

        Raises:
            ValueError: If no molecule provided and none cached
        """
        if mol_or_smiles is not None:
            # Use provided molecule (static mode)
            return self._core._prepare_mol(mol_or_smiles, self.add_hs)
        elif self._mol is not None:
            # Use cached molecule (instance mode)
            return self._mol
        else:
            raise ValueError("No molecule provided. Initialize with a molecule or pass one to the method.")

    def get_feature(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> Dict[str, torch.Tensor]:
        """
        Get all molecular features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)

        Returns:
            Dictionary containing all molecular features
        """
        # Check if using cached molecule and features already computed
        if mol_or_smiles is None and self._mol is not None and 'features' in self._cache:
            return self._cache['features']

        mol = self._get_mol(mol_or_smiles)
        features = self._core.get_feature(mol)

        # Cache if using instance mode
        if mol_or_smiles is None and self._mol is not None:
            self._cache['features'] = features

        return features

    def get_graph(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
                  add_hs: Optional[bool] = None) -> Tuple[Dict, Dict]:
        """
        Get graph representation (node and edge features).

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)
            add_hs: Whether to add hydrogens (overrides instance setting)

        Returns:
            Tuple of (node_dict, edge_dict)
        """
        # Check if using cached molecule and graph already computed
        cache_key = f'graph_hs_{add_hs if add_hs is not None else self.add_hs}'
        if mol_or_smiles is None and self._mol is not None and cache_key in self._cache:
            return self._cache[cache_key]

        # Determine add_hs setting
        use_add_hs = add_hs if add_hs is not None else self.add_hs

        # Get molecule
        if mol_or_smiles is not None:
            # Prepare new molecule with specified add_hs
            mol = self._core._prepare_mol(mol_or_smiles, use_add_hs)
        else:
            # Use cached molecule (may need to re-prepare if add_hs changed)
            if add_hs is not None and add_hs != self.add_hs:
                # Re-prepare with different hydrogen setting
                if isinstance(self._mol, Chem.Mol):
                    mol = self._core._prepare_mol(self._mol, use_add_hs)
                else:
                    mol = self._mol
            else:
                mol = self._mol

        graph = self._core.get_graph(mol, add_hs=False)  # Already prepared

        # Cache if using instance mode
        if mol_or_smiles is None and self._mol is not None:
            self._cache[cache_key] = graph

        return graph

    def get_fingerprints(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> Dict[str, torch.Tensor]:
        """
        Get molecular fingerprints.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)

        Returns:
            Dictionary containing various fingerprints
        """
        if mol_or_smiles is None and self._mol is not None and 'fingerprints' in self._cache:
            return self._cache['fingerprints']

        mol = self._get_mol(mol_or_smiles)
        fingerprints = self._core.get_fingerprints(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache['fingerprints'] = fingerprints

        return fingerprints

    def get_physicochemical_features(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> torch.Tensor:
        """
        Get physicochemical molecular features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)

        Returns:
            Tensor of physicochemical features
        """
        if mol_or_smiles is None and self._mol is not None and 'physicochemical' in self._cache:
            return self._cache['physicochemical']

        mol = self._get_mol(mol_or_smiles)
        features = self._core.get_physicochemical_features(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache['physicochemical'] = features

        return features

    def get_druglike_features(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> torch.Tensor:
        """
        Get drug-likeness features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)

        Returns:
            Tensor of drug-likeness features
        """
        if mol_or_smiles is None and self._mol is not None and 'druglike' in self._cache:
            return self._cache['druglike']

        mol = self._get_mol(mol_or_smiles)
        features = self._core.get_druglike_features(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache['druglike'] = features

        return features

    def get_structural_features(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None) -> torch.Tensor:
        """
        Get structural features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)

        Returns:
            Tensor of structural features
        """
        if mol_or_smiles is None and self._mol is not None and 'structural' in self._cache:
            return self._cache['structural']

        mol = self._get_mol(mol_or_smiles)
        features = self._core.get_structural_features(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache['structural'] = features

        return features

    def get_atom_features(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
                         add_hs: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)
            add_hs: Whether to add hydrogens (overrides instance setting)

        Returns:
            Tuple of (atom_features, coordinates)
        """
        cache_key = f'atom_features_hs_{add_hs if add_hs is not None else self.add_hs}'
        if mol_or_smiles is None and self._mol is not None and cache_key in self._cache:
            return self._cache[cache_key]

        # Get molecule with appropriate hydrogen setting
        use_add_hs = add_hs if add_hs is not None else self.add_hs
        if mol_or_smiles is not None:
            mol = self._core._prepare_mol(mol_or_smiles, use_add_hs)
        else:
            if add_hs is not None and add_hs != self.add_hs:
                mol = self._core._prepare_mol(self._mol, use_add_hs)
            else:
                mol = self._mol

        features = self._core.get_atom_features(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache[cache_key] = features

        return features

    def get_bond_features(self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
                         add_hs: Optional[bool] = None) -> torch.Tensor:
        """
        Get bond-level features.

        Args:
            mol_or_smiles: Optional molecule (uses cached if not provided)
            add_hs: Whether to add hydrogens (overrides instance setting)

        Returns:
            Bond features tensor
        """
        cache_key = f'bond_features_hs_{add_hs if add_hs is not None else self.add_hs}'
        if mol_or_smiles is None and self._mol is not None and cache_key in self._cache:
            return self._cache[cache_key]

        # Get molecule with appropriate hydrogen setting
        use_add_hs = add_hs if add_hs is not None else self.add_hs
        if mol_or_smiles is not None:
            mol = self._core._prepare_mol(mol_or_smiles, use_add_hs)
        else:
            if add_hs is not None and add_hs != self.add_hs:
                mol = self._core._prepare_mol(self._mol, use_add_hs)
            else:
                mol = self._mol

        features = self._core.get_bond_features(mol)

        if mol_or_smiles is None and self._mol is not None:
            self._cache[cache_key] = features

        return features

    def clear_cache(self):
        """Clear all cached features (keeps molecule)."""
        self._cache = {}
        if self._mol is not None:
            self._parse_molecule()

    def set_molecule(self, mol_or_smiles: Union[str, Chem.Mol], add_hs: Optional[bool] = None):
        """
        Set a new molecule and clear cache.

        Args:
            mol_or_smiles: New molecule (RDKit mol or SMILES)
            add_hs: Whether to add hydrogens (uses instance default if not specified)
        """
        use_add_hs = add_hs if add_hs is not None else self.add_hs
        self._mol = self._core._prepare_mol(mol_or_smiles, use_add_hs)
        self._cache = {}
        self._parse_molecule()
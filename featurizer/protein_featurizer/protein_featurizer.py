"""
Efficient Protein Featurizer with one-time parsing.

This module provides a high-level API for protein feature extraction
with efficient caching of parsed PDB data.
"""

import os
import tempfile
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np

from .pdb_standardizer import PDBStandardizer
from .residue_featurizer import ResidueFeaturizer


class ProteinFeaturizer:
    """
    Efficient protein featurizer that parses PDB once and caches results.

    Examples:
        >>> # Parse once, extract multiple features efficiently
        >>> featurizer = ProteinFeaturizer("protein.pdb")
        >>> sequence = featurizer.get_sequence_features()
        >>> geometry = featurizer.get_geometric_features()
        >>> sasa = featurizer.get_sasa_features()
    """

    def __init__(self, pdb_file: str, standardize: bool = True,
                 keep_hydrogens: bool = False):
        """
        Initialize and parse PDB file once.

        Args:
            pdb_file: Path to PDB file
            standardize: Whether to standardize the PDB first
            keep_hydrogens: Whether to keep hydrogens during standardization
        """
        self.input_file = pdb_file
        self.standardize = standardize
        self.keep_hydrogens = keep_hydrogens

        # Check if file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        # Standardize if requested
        if standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                self.tmp_pdb = tmp_file.name

            standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)
            standardizer.standardize(pdb_file, self.tmp_pdb)
            pdb_to_process = self.tmp_pdb
        else:
            self.tmp_pdb = None
            pdb_to_process = pdb_file

        # Parse PDB once
        self._featurizer = ResidueFeaturizer(pdb_to_process)
        self._parse_structure()

        # Cache for computed features
        self._cache = {}

    def _parse_structure(self):
        """Parse structure and cache basic data."""
        # Get residues
        self.residues = self._featurizer.get_residues()
        self.num_residues = len(self.residues)

        # Build coordinate tensor
        self.coords = torch.zeros(self.num_residues, 15, 3)
        self.residue_types = torch.from_numpy(
            np.array(self.residues)[:, 2].astype(int)
        )

        for idx, residue in enumerate(self.residues):
            residue_coord = torch.as_tensor(
                self._featurizer.get_residue_coordinates(residue).tolist()
            )
            self.coords[idx, :residue_coord.shape[0], :] = residue_coord
            # Sidechain centroid
            self.coords[idx, -1, :] = residue_coord[4:, :].mean(0)

        # Extract CA and SC coordinates
        self.coords_CA = self.coords[:, 1:2, :]
        self.coords_SC = self.coords[:, -1:, :]
        self.coord = torch.cat([self.coords_CA, self.coords_SC], dim=1)

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'tmp_pdb') and self.tmp_pdb and os.path.exists(self.tmp_pdb):
            os.unlink(self.tmp_pdb)

    def get_sequence_features(self) -> Dict[str, Any]:
        """
        Get amino acid sequence and position features.

        Returns:
            Dictionary with residue types and one-hot encoding
        """
        if 'sequence' not in self._cache:
            residue_one_hot = torch.nn.functional.one_hot(
                self.residue_types, num_classes=21
            )

            self._cache['sequence'] = {
                'residue_types': self.residue_types,
                'residue_one_hot': residue_one_hot,
                'num_residues': self.num_residues
            }

        return self._cache['sequence']

    def get_geometric_features(self) -> Dict[str, Any]:
        """
        Get geometric features including distances, angles, and dihedrals.

        Returns:
            Dictionary with geometric measurements
        """
        if 'geometric' not in self._cache:
            # Get geometric features
            dihedrals, has_chi = self._featurizer.get_dihedral_angles(
                self.coords, self.residue_types
            )
            terminal_flags = self.get_terminal_flags()
            curvature = self._featurizer._calculate_backbone_curvature(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            torsion = self._featurizer._calculate_backbone_torsion(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            self_distance, self_vector = self._featurizer._calculate_self_distances_vectors(
                self.coords
            )

            self._cache['geometric'] = {
                'dihedrals': dihedrals,
                'has_chi_angles': has_chi,
                'backbone_curvature': curvature,
                'backbone_torsion': torsion,
                'self_distances': self_distance,
                'self_vectors': self_vector,
                'coordinates': self.coords
            }

        return self._cache['geometric']

    def get_sasa_features(self) -> torch.Tensor:
        """
        Get Solvent Accessible Surface Area features.

        Returns:
            SASA tensor with multiple components per residue
        """
        if 'sasa' not in self._cache:
            self._cache['sasa'] = self._featurizer.calculate_sasa()

        return self._cache['sasa']

    def get_contact_map(self, cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get residue-residue contact map and distances.

        Args:
            cutoff: Distance cutoff for contacts (default: 8.0 Å)

        Returns:
            Dictionary with contact information
        """
        cache_key = f'contact_map_{cutoff}'

        if cache_key not in self._cache:
            distance_adj, adj, vectors = self._featurizer._calculate_interaction_features(
                self.coords, cutoff=cutoff
            )

            # Get sparse representation
            sparse = distance_adj.to_sparse(sparse_dim=2)
            src, dst = sparse.indices()
            distances = sparse.values()

            self._cache[cache_key] = {
                'adjacency_matrix': adj,
                'distance_matrix': distance_adj,
                'edges': (src, dst),
                'edge_distances': distances,
                'interaction_vectors': vectors
            }

        return self._cache[cache_key]

    def get_relative_position(self, cutoff: int = 32) -> torch.Tensor:
        """
        Get relative position encoding between residues.

        Args:
            cutoff: Maximum relative position to consider

        Returns:
            One-hot encoded relative position tensor
        """
        cache_key = f'relative_position_{cutoff}'

        if cache_key not in self._cache:
            self._cache[cache_key] = self._featurizer.get_relative_position(
                cutoff=cutoff, onehot=True
            )

        return self._cache[cache_key]

    def get_node_features(self) -> Dict[str, Any]:
        """
        Get all node (residue-level) features.

        Returns:
            Dictionary with scalar and vector node features
        """
        if 'node_features' not in self._cache:
            scalar_features, vector_features = self._featurizer._extract_residue_features(
                self.coords, self.residue_types
            )

            self._cache['node_features'] = {
                'coordinates': self.coord,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache['node_features']

    def get_edge_features(self, distance_cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get all edge (interaction) features.

        Args:
            distance_cutoff: Distance cutoff for interactions

        Returns:
            Dictionary with edge indices and features
        """
        cache_key = f'edge_features_{distance_cutoff}'

        if cache_key not in self._cache:
            edges, scalar_features, vector_features = \
                self._featurizer._extract_interaction_features(
                    self.coords, distance_cutoff=distance_cutoff
                )

            self._cache[cache_key] = {
                'edges': edges,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache[cache_key]

    def get_terminal_flags(self) -> Dict[str, torch.Tensor]:
        """
        Get N-terminal and C-terminal residue flags.

        Returns:
            Dictionary with terminal flags
        """
        if 'terminal_flags' not in self._cache:
            n_terminal, c_terminal = self._featurizer.get_terminal_flags()
            self._cache['terminal_flags'] = {
                'n_terminal': n_terminal,
                'c_terminal': c_terminal
            }

        return self._cache['terminal_flags']

    def get_all_features(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all features at once.

        Args:
            save_to: Optional path to save features

        Returns:
            Dictionary containing all features
        """
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()

        features = {
            'node': node_features,
            'edge': edge_features,
            'metadata': {
                'input_file': self.input_file,
                'standardized': self.standardize,
                'hydrogens_removed': not self.keep_hydrogens if self.standardize else None,
                'num_residues': self.num_residues
            }
        }

        if save_to:
            torch.save(features, save_to)

        return features

    # Alias for backward compatibility
    extract = get_all_features
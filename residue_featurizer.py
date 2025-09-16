"""
Residue Featurizer Module

This module provides functionality to extract structural features from protein PDB files
for machine learning applications. It includes geometric features, SASA calculations,
and graph-based interaction features.
"""

import os
import sys
import warnings
import contextlib
from io import StringIO
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import freesasa as fs


# Amino acid mappings
AMINO_ACID_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

AMINO_ACID_1TO3 = {v: k for k, v in AMINO_ACID_3TO1.items()}
AMINO_ACID_1_TO_INT = {k: i for i, k in enumerate(sorted(AMINO_ACID_1TO3.keys()))}
AMINO_ACID_3_TO_INT = {AMINO_ACID_1TO3[k]: i for i, k in enumerate(sorted(AMINO_ACID_1TO3.keys()))}

# Add unknown residue mapping
AMINO_ACID_1_TO_INT['X'] = 20
AMINO_ACID_3_TO_INT['UNK'] = 20


@contextlib.contextmanager
def suppress_freesasa_warnings():
    """
    Context manager to suppress FreeSASA warnings about unknown atoms.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            yield
        finally:
            sys.stderr = old_stderr


class ResidueFeaturizer:
    """
    A class for extracting structural features from protein PDB files.

    This class provides methods to compute various features including:
    - Geometric features (distances, angles, dihedrals)
    - Solvent accessible surface area (SASA)
    - Residue-level and interaction features
    - Graph representations of protein structure
    """

    def __init__(self, pdb_file: str):
        """
        Initialize the featurizer with a PDB file.

        Args:
            pdb_file: Path to the PDB file
        """
        self.pdb_file = pdb_file
        self.protein_indices, self.hetero_indices, self.protein_atom_info, self.hetero_atom_info = \
            self._parse_pdb(pdb_file)

    def _parse_pdb(self, pdb_file: str) -> Tuple[pd.MultiIndex, pd.MultiIndex, pd.DataFrame, pd.DataFrame]:
        """
        Parse PDB file and extract atom information.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (protein_indices, hetero_indices, protein_atom_info, hetero_atom_info)
        """
        with open(pdb_file, 'r') as f:
            lines = f.read().split('\n')

        protein_index, hetero_index = [], []
        protein_data, hetero_data = {'coord': []}, {'coord': []}

        for line in lines:
            record_type = line[:6].strip()
            if record_type in ['ATOM', 'HETATM'] and len(line) > 13 and line[13] != 'H' and line[17:20].strip() != 'HOH':
                atom_type = line[12:17].strip()
                res_type = AMINO_ACID_3_TO_INT.get(line[17:20].strip(), 20)
                chain_id = line[21]
                res_num = int(line[22:26])
                xyz = [float(line[idx:idx + 8]) for idx in range(30, 54, 8)]

                if record_type == 'ATOM':
                    protein_index.append((chain_id, res_num, res_type, atom_type))
                    protein_data['coord'].append(xyz)
                elif record_type == 'HETATM':
                    hetero_index.append(('HETERO', res_num, res_type, atom_type))
                    hetero_data['coord'].append(xyz)

        protein_index = pd.MultiIndex.from_tuples(protein_index, names=['chain', 'res_num', 'AA', 'atom'])
        hetero_index = pd.MultiIndex.from_tuples(hetero_index, names=['chain', 'res_num', 'AA', 'atom'])

        protein_atom_info = pd.DataFrame(protein_data, index=protein_index)
        hetero_atom_info = pd.DataFrame(hetero_data, index=hetero_index)

        return protein_index, hetero_index, protein_atom_info, hetero_atom_info

    def get_residues(self) -> List[Tuple]:
        """
        Get list of all protein residues.

        Returns:
            List of (chain, residue_number, residue_type) tuples
        """
        return sorted(set(list([(chain, num, res) for chain, num, res, atom in self.protein_indices])))

    def get_residue_coordinates(self, residue_index: Tuple) -> pd.Series:
        """
        Get coordinates for a specific residue.

        Args:
            residue_index: Tuple of (chain, residue_number, residue_type)

        Returns:
            Coordinates of all atoms in the residue
        """
        return self.protein_atom_info.coord.xs(residue_index)

    def get_terminal_flags(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify N-terminal and C-terminal residues.

        Returns:
            Tuple of (n_terminal_flags, c_terminal_flags) as boolean tensors
        """
        residues = self.get_residues()

        residue_array = np.array(residues, dtype=object)
        chains = residue_array[:, 0]
        res_nums = residue_array[:, 1].astype(int)

        unique_chains = np.unique(chains)
        n_terminal = np.zeros(len(residues), dtype=bool)
        c_terminal = np.zeros(len(residues), dtype=bool)

        chain_masks = chains[:, None] == unique_chains[None, :]

        for i, chain in enumerate(unique_chains):
            mask = chain_masks[:, i]
            chain_indices = np.where(mask)[0]
            chain_res_nums = res_nums[mask]

            min_idx = chain_indices[np.argmin(chain_res_nums)]
            max_idx = chain_indices[np.argmax(chain_res_nums)]

            n_terminal[min_idx] = True
            c_terminal[max_idx] = True

        return torch.tensor(n_terminal, dtype=torch.bool), torch.tensor(c_terminal, dtype=torch.bool)

    def get_relative_position(self, cutoff: int = 32, onehot: bool = True) -> torch.Tensor:
        """
        Calculate relative position encoding for residue pairs.

        Args:
            cutoff: Maximum relative position to consider
            onehot: Whether to return one-hot encoded positions

        Returns:
            Relative position tensor
        """
        residues = self.get_residues()
        num_residues = len(residues)

        chain_indices = {}
        for idx, (chain, num, res) in enumerate(residues):
            if chain not in chain_indices:
                chain_indices[chain] = []
            chain_indices[chain].append(idx)

        relative_positions = torch.full((num_residues, num_residues), -1, dtype=torch.long)

        for chain, indices in chain_indices.items():
            if len(indices) <= 1:
                continue

            indices_tensor = torch.tensor(indices, dtype=torch.long)
            num_chain_residues = len(indices)

            arrange = torch.arange(num_chain_residues, dtype=torch.long)
            chain_relative_positions = (arrange[:, None] - arrange[None, :]).abs()
            chain_relative_positions = torch.clamp(chain_relative_positions, max=cutoff+1)
            chain_relative_positions = torch.where(chain_relative_positions > cutoff, 33, chain_relative_positions)

            relative_positions[indices_tensor[:, None], indices_tensor[None, :]] = chain_relative_positions

        if onehot:
            relative_positions_mapped = torch.where(relative_positions == -1, 34, relative_positions)
            relative_positions_onehot = F.one_hot(relative_positions_mapped, num_classes=35).float()
            return relative_positions_onehot

        return relative_positions

    def calculate_sasa(self) -> torch.Tensor:
        """
        Calculate Solvent Accessible Surface Area (SASA) for each residue.

        Returns:
            SASA features tensor
        """
        with suppress_freesasa_warnings():
            sasas = [
                [
                    values.total / 350,
                    values.polar / 350,
                    values.apolar / 350,
                    values.mainChain / 350,
                    values.sideChain / 350,
                    values.relativeTotal,
                    values.relativePolar,
                    values.relativeApolar,
                    values.relativeMainChain,
                    values.relativeSideChain
                ]
                for chain, residues in fs.calc(fs.Structure(self.pdb_file)).residueAreas().items()
                for residue, values in residues.items()
            ]

        return torch.nan_to_num(torch.as_tensor(sasas))

    def _calculate_dihedral(self, coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate dihedral angles from coordinates.

        Args:
            coords: Coordinate tensor
            eps: Small value for numerical stability

        Returns:
            Dihedral angles tensor
        """
        shape = coords.shape
        coords_flat = coords.reshape(shape[0] * shape[1], shape[2])

        U = F.normalize(coords_flat[1:, :] - coords_flat[:-1, :], dim=-1)
        u_2 = U[:-2, :]
        u_1 = U[1:-1, :]
        u_0 = U[2:, :]

        n_2 = F.normalize(torch.cross(u_2, u_1, dim=1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=1), dim=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)

        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1, 2), 'constant', 0)

        return D.view((int(D.size(0)/shape[1]), shape[1]))

    def get_dihedral_angles(self, coords: torch.Tensor, res_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate backbone and sidechain dihedral angles.

        Args:
            coords: Residue coordinates tensor (num_residues, 15, 3)
            res_types: Residue type indices tensor (num_residues,)

        Returns:
            Tuple of (dihedral_angles, has_chi_angles)
        """
        # Chi angle residue indices
        chi_indices = {
            'chi1': torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            'chi2': torch.tensor([2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19]),
            'chi3': torch.tensor([3, 8, 10, 13, 14]),
            'chi4': torch.tensor([8, 14]),
            'chi5': torch.tensor([14])
        }

        is_ILE = torch.isin(res_types, torch.tensor([7])).int().unsqueeze(1).unsqueeze(2)
        is_not_ILE = 1 - is_ILE

        has_chi = torch.stack([torch.isin(res_types, chi_indices[f'chi{i}']).int() for i in range(1, 6)], dim=1)

        # Backbone dihedrals
        N_CA_C = coords[:, :3, :]
        backbone_dihedrals = self._calculate_dihedral(N_CA_C)

        # Sidechain dihedrals
        N_A_B_G_D_E_Z_ILE = torch.cat([coords[:, :2, :], coords[:, 4:6, :], coords[:, 7:11, :]], dim=1) * is_ILE
        N_A_B_G_D_E_Z_no_ILE = torch.cat([coords[:, :2, :], coords[:, 4:10, :]], dim=1) * is_not_ILE
        N_A_B_G_D_E_Z = N_A_B_G_D_E_Z_ILE + N_A_B_G_D_E_Z_no_ILE

        side_chain_dihedrals = self._calculate_dihedral(N_A_B_G_D_E_Z)[:, 1:-2] * has_chi

        dihedrals = torch.cat([backbone_dihedrals, side_chain_dihedrals], dim=1)

        return dihedrals, has_chi

    def _calculate_local_frames(self, coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate local coordinate frames for each residue.

        Args:
            coords: Residue coordinates
            eps: Small value for numerical stability

        Returns:
            Local frames tensor
        """
        p_N, p_Ca, p_C = coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]

        u = p_N - p_Ca
        v = p_C - p_Ca

        x_axis = F.normalize(u, dim=-1, eps=eps)
        z_axis = F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=eps)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)

        return torch.stack([x_axis, y_axis, z_axis], dim=2)

    def _calculate_backbone_curvature(self, coords: torch.Tensor, terminal_flags: Tuple[torch.Tensor, torch.Tensor],
                                     eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate backbone curvature.

        Args:
            coords: Residue coordinates
            terminal_flags: N and C terminal flags
            eps: Small value for numerical stability

        Returns:
            Backbone curvature tensor
        """
        ca_coords = coords[:, 1, :]

        p_im1 = ca_coords[:-2]
        p_i = ca_coords[1:-1]
        p_ip1 = ca_coords[2:]

        v1 = p_im1 - p_i
        v2 = p_ip1 - p_i

        cos_theta = (F.normalize(v1, dim=-1, eps=eps) * F.normalize(v2, dim=-1, eps=eps)).sum(dim=-1)
        curvature_rad = torch.acos(torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps))

        curvature_rad = F.pad(curvature_rad, (1, 1), 'constant', 0)
        n_terminal, c_terminal = terminal_flags
        curvature_rad = curvature_rad * ~n_terminal
        curvature_rad = curvature_rad * ~c_terminal

        return curvature_rad

    def _calculate_backbone_torsion(self, coords: torch.Tensor, terminal_flags: Tuple[torch.Tensor, torch.Tensor],
                                   eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate backbone torsion.

        Args:
            coords: Residue coordinates
            terminal_flags: N and C terminal flags
            eps: Small value for numerical stability

        Returns:
            Backbone torsion tensor
        """
        ca_coords = coords[:, 1, :]

        p0 = ca_coords[:-3]
        p1 = ca_coords[1:-2]
        p2 = ca_coords[2:-1]
        p3 = ca_coords[3:]

        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = F.normalize(torch.cross(b1, b2, dim=-1), dim=-1, eps=eps)
        n2 = F.normalize(torch.cross(b2, b3, dim=-1), dim=-1, eps=eps)

        x = (n1 * n2).sum(dim=-1)
        y = (torch.cross(n1, n2, dim=-1) * F.normalize(b2, dim=-1, eps=eps)).sum(dim=-1)
        torsion_rad = torch.atan2(y, x)

        torsion_rad = F.pad(torsion_rad, (1, 2), 'constant', 0)
        n_terminal, c_terminal = terminal_flags
        torsion_rad = torsion_rad * ~n_terminal
        torsion_rad = torsion_rad * ~c_terminal

        return torsion_rad

    def _calculate_self_distances_vectors(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate intra-residue distances and vectors.

        Args:
            coords: Residue coordinates

        Returns:
            Tuple of (distances, vectors)
        """
        coords_subset = torch.cat([coords[:, :4, :], coords[:, -1:, :]], dim=1)

        distance = torch.cdist(coords_subset, coords_subset)
        mask_sca = torch.triu(torch.ones_like(distance), diagonal=1).bool()
        distance = torch.masked_select(distance, mask_sca).view(distance.shape[0], -1)

        vectors = coords_subset[:, None] - coords_subset[:, :, None]
        vectors = vectors.view(coords.shape[0], 25, 3)
        vectors = torch.index_select(vectors, 1, torch.tensor([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]))

        return torch.nan_to_num(distance), torch.nan_to_num(vectors)

    def _calculate_forward_reverse(self, coord: torch.Tensor, terminal_flags: Tuple[torch.Tensor, torch.Tensor]) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculate forward and reverse residue connection features.

        Args:
            coord: Residue coordinates
            terminal_flags: N and C terminal flags

        Returns:
            Tuple of ((forward_vectors, forward_distances), (reverse_vectors, reverse_distances))
        """
        ca_coords = coord[:, 1, :]  # CA coordinates
        sc_coords = coord[:, -1, :]  # SC coordinates

        n_terminal, c_terminal = terminal_flags

        forward_vector = torch.zeros(coord.shape[0], 4, 3)
        forward_distance = torch.zeros(coord.shape[0], 4)
        reverse_vector = torch.zeros(coord.shape[0], 4, 3)
        reverse_distance = torch.zeros(coord.shape[0], 4)

        if coord.shape[0] > 1:
            ca_diff = ca_coords[1:] - ca_coords[:-1]
            sc_diff = sc_coords[1:] - sc_coords[:-1]
            ca_sc_diff = sc_coords[1:] - ca_coords[:-1]
            sc_ca_diff = ca_coords[1:] - sc_coords[:-1]

            forward_vector[:-1] = torch.stack([ca_diff, sc_diff, ca_sc_diff, sc_ca_diff], dim=1)
            forward_distance[:-1] = torch.norm(forward_vector[:-1], dim=-1)

            c_mask = ~c_terminal[:-1]
            forward_vector[:-1] *= c_mask[:, None, None]
            forward_distance[:-1] *= c_mask[:, None]

            reverse_vector[1:] = torch.stack([-ca_diff, -sc_diff, ca_coords[:-1] - sc_coords[1:],
                                             sc_coords[:-1] - ca_coords[1:]], dim=1)
            reverse_distance[1:] = torch.norm(reverse_vector[1:], dim=-1)

            n_mask = (~n_terminal[1:])
            reverse_vector[1:] *= n_mask[:, None, None]
            reverse_distance[1:] *= n_mask[:, None]

        forward_vector = torch.nan_to_num(forward_vector)
        reverse_vector = torch.nan_to_num(reverse_vector)
        forward_distance = torch.nan_to_num(forward_distance)
        reverse_distance = torch.nan_to_num(reverse_distance)

        return (forward_vector, forward_distance), (reverse_vector, reverse_distance)

    def _calculate_interaction_features(self, coords: torch.Tensor, cutoff: float = 8) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate inter-residue interaction features.

        Args:
            coords: Residue coordinates
            cutoff: Distance cutoff for interactions

        Returns:
            Tuple of (distances, adjacency_matrix, interaction_vectors)
        """
        coord_CA = coords[:, 1:2, :].transpose(0, 1)
        coord_SC = coords[:, -1:, :].transpose(0, 1)
        mask = (1 - torch.eye(coords.shape[0])).int()

        dm_CA_CA = torch.cdist(coord_CA, coord_CA)[0]
        dm_SC_SC = torch.cdist(coord_SC, coord_SC)[0]
        dm_CA_SC = torch.cdist(coord_CA, coord_SC)[0]
        dm_SC_CA = torch.cdist(coord_SC, coord_CA)[0]

        adj_CA_CA = (dm_CA_CA < cutoff) * mask
        adj_SC_SC = (dm_SC_SC < cutoff) * mask
        adj_CA_SC = (dm_CA_SC < cutoff) * mask
        adj_SC_CA = (dm_SC_CA < cutoff) * mask

        adj = adj_CA_CA | adj_SC_SC | adj_CA_SC | adj_SC_CA

        dm_all = torch.stack((dm_CA_CA, dm_SC_SC, dm_CA_SC, dm_SC_CA), dim=-1)
        dm_select = dm_all * adj[:, :, None]

        # Calculate interaction vectors
        coord_CA_SC = torch.cat([coords[:, 1:2, :], coords[:, -1:, :]], dim=1)
        coord_SC_CA = torch.cat([coords[:, -1:, :], coords[:, 1:2, :]], dim=1)

        vector1 = coord_CA_SC[:, None, :] - coord_CA_SC[:, :, :]
        vector3 = coord_CA_SC[:, None, :] - coord_SC_CA[:, :, :]
        vectors = torch.cat([vector1, -vector1, vector3, -vector3], dim=2).nan_to_num()
        vectors = vectors * adj[:, :, None, None]

        return torch.nan_to_num(dm_select), adj, vectors

    def _extract_residue_features(self, coords: torch.Tensor, residue_types: torch.Tensor) -> \
            Tuple[Tuple, Tuple]:
        """
        Extract all residue-level features.

        Args:
            coords: Residue coordinates
            residue_types: Residue type indices

        Returns:
            Tuple of (scalar_features, vector_features)
        """
        # One-hot encoding of residue types
        residue_one_hot = F.one_hot(residue_types, num_classes=21)
        terminal_flags = self.get_terminal_flags()

        # Local self features
        self_distance, self_vector = self._calculate_self_distances_vectors(coords)

        # Local frames
        local_frames = self._calculate_local_frames(coords)

        # Dihedral angles and curvature
        dihedrals, has_chi_angles = self.get_dihedral_angles(coords, residue_types)
        backbone_curvature = self._calculate_backbone_curvature(coords, terminal_flags)
        backbone_torsion = self._calculate_backbone_torsion(coords, terminal_flags)

        degree = torch.cat([dihedrals, backbone_curvature[:, None], backbone_torsion[:, None]], dim=1)
        degree_feature = torch.cat([torch.cos(degree), torch.sin(degree)], dim=1)

        # SASA features
        sasa = self.calculate_sasa()

        # Forward/reverse features
        forward, reverse = self._calculate_forward_reverse(coords, terminal_flags)
        forward_vector, forward_distance = forward
        reverse_vector, reverse_distance = reverse

        rf_vector = torch.cat([forward_vector, reverse_vector], dim=1)
        rf_distance = torch.cat([forward_distance, reverse_distance], dim=1)

        # Collect all features
        scalar_features = (
            residue_one_hot,
            torch.stack(terminal_flags, dim=1),
            self_distance,
            degree_feature,
            has_chi_angles,
            sasa,
            rf_distance,
        )

        vector_features = (
            self_vector,
            rf_vector,
            local_frames,
        )

        return scalar_features, vector_features

    def _extract_interaction_features(self, coords: torch.Tensor, distance_cutoff: float = 8,
                                     relative_position_cutoff: int = 32) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple, Tuple]:
        """
        Extract interaction features between residues.

        Args:
            coords: Residue coordinates
            distance_cutoff: Distance cutoff for interactions
            relative_position_cutoff: Cutoff for relative position encoding

        Returns:
            Tuple of (edges, scalar_features, vector_features)
        """
        relative_position = self.get_relative_position(cutoff=relative_position_cutoff, onehot=True)
        distance_adj, adj, interaction_vectors = self._calculate_interaction_features(coords, cutoff=distance_cutoff)

        # Convert to sparse format
        sparse = distance_adj.to_sparse(sparse_dim=2)
        src, dst = sparse.indices()
        distance = sparse.values()

        relative_position = relative_position[src, dst]
        vectors = interaction_vectors[src, dst, :]

        edges = (src, dst)
        edge_scalar_features = (distance, relative_position)
        edge_vector_features = (vectors,)

        return edges, edge_scalar_features, edge_vector_features

    def get_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract all features for the protein.

        Returns:
            Tuple of (node_features, edge_features) dictionaries
        """
        residues = self.get_residues()
        coords = torch.zeros(len(residues), 15, 3)
        residue_types = torch.from_numpy(np.array(residues)[:, 2].astype(int))

        # Build coordinate tensor
        for idx, residue in enumerate(residues):
            residue_coord = torch.as_tensor(self.get_residue_coordinates(residue).tolist())
            coords[idx, :residue_coord.shape[0], :] = residue_coord
            coords[idx, -1, :] = residue_coord[4:, :].mean(0)  # Sidechain centroid

        # Extract CA and SC coordinates
        coords_CA = coords[:, 1:2, :]
        coords_SC = coords[:, -1:, :]
        coord = torch.cat([coords_CA, coords_SC], dim=1)

        # Extract features
        node_scalar_features, node_vector_features = self._extract_residue_features(coords, residue_types)
        edges, edge_scalar_features, edge_vector_features = self._extract_interaction_features(coords)

        # Package features
        node = {
            'coord': coord,
            'node_scalar_features': node_scalar_features,
            'node_vector_features': node_vector_features
        }

        edge = {
            'edges': edges,
            'edge_scalar_features': edge_scalar_features,
            'edge_vector_features': edge_vector_features
        }

        return node, edge


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Residue Feature Extractor for Proteins')
    parser.add_argument('input_pdb', help='Input PDB file path')
    parser.add_argument('--output', '-o', help='Output file path for features (default: features.pt)',
                       default='features.pt')

    args = parser.parse_args()

    try:
        print(f"Processing {args.input_pdb}...")
        featurizer = ResidueFeaturizer(args.input_pdb)
        node, edge = featurizer.get_features()

        # Combine features
        features = {'node': node, 'edge': edge}

        # Save features
        torch.save(features, args.output)
        print(f"Features saved to {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
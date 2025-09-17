"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import freesasa

####################################################################################################
################################       PROTEIN      ################################################
####################################################################################################

amino_acid_mapping = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                      'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                      'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

amino_acid_mapping_reverse = {v: k for k, v in amino_acid_mapping.items()}
amino_acid_3_to_int = {amino_acid_mapping_reverse[k]: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}
amino_acid_1_to_int = {k: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}

aa_letter = list(amino_acid_mapping.keys())

res_token = {
    'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4,
    'GLN': 5,  'GLU': 6,  'GLY': 7,  'HIS': 8,  'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'XXX': 20, 'METAL': 21,
}

res_atm_token = {
    ('ALA', 'C'): 0, ('ALA', 'CA'): 1, ('ALA', 'CB'): 2, ('ALA', 'N'): 3, ('ALA', 'O'): 4,
    ('ARG', 'C'): 5, ('ARG', 'CA'): 6, ('ARG', 'CB'): 7, ('ARG', 'CD'): 8, ('ARG', 'CG'): 9, ('ARG', 'CZ'): 10, ('ARG', 'N'): 11, ('ARG', 'NE'): 12, ('ARG', 'NH1'): 13, ('ARG', 'NH2'): 14, ('ARG', 'O'): 15,
    ('ASN', 'C'): 16, ('ASN', 'CA'): 17, ('ASN', 'CB'): 18, ('ASN', 'CG'): 19, ('ASN', 'N'): 20, ('ASN', 'ND2'): 21, ('ASN', 'O'): 22, ('ASN', 'OD1'): 23,
    ('ASP', 'C'): 24, ('ASP', 'CA'): 25, ('ASP', 'CB'): 26, ('ASP', 'CG'): 27, ('ASP', 'N'): 28, ('ASP', 'O'): 29, ('ASP', 'OD1'): 30, ('ASP', 'OD2'): 31,
    ('CYS', 'C'): 32, ('CYS', 'CA'): 33, ('CYS', 'CB'): 34, ('CYS', 'N'): 35, ('CYS', 'O'): 36, ('CYS', 'SG'): 37,
    ('GLN', 'C'): 38, ('GLN', 'CA'): 39, ('GLN', 'CB'): 40, ('GLN', 'CD'): 41, ('GLN', 'CG'): 42, ('GLN', 'N'): 43, ('GLN', 'NE2'): 44, ('GLN', 'O'): 45, ('GLN', 'OE1'): 46,
    ('GLU', 'C'): 47, ('GLU', 'CA'): 48, ('GLU', 'CB'): 49, ('GLU', 'CD'): 50, ('GLU', 'CG'): 51, ('GLU', 'N'): 52, ('GLU', 'O'): 53, ('GLU', 'OE1'): 54, ('GLU', 'OE2'): 55,
    ('GLY', 'C'): 56, ('GLY', 'CA'): 57, ('GLY', 'N'): 58, ('GLY', 'O'): 59,
    ('HIS', 'C'): 60, ('HIS', 'CA'): 61, ('HIS', 'CB'): 62, ('HIS', 'CD2'): 63, ('HIS', 'CE1'): 64, ('HIS', 'CG'): 65, ('HIS', 'N'): 66, ('HIS', 'ND1'): 67, ('HIS', 'NE2'): 68, ('HIS', 'O'): 69,
    ('ILE', 'C'): 70, ('ILE', 'CA'): 71, ('ILE', 'CB'): 72, ('ILE', 'CD1'): 73, ('ILE', 'CG1'): 74, ('ILE', 'CG2'): 75, ('ILE', 'N'): 76, ('ILE', 'O'): 77,
    ('LEU', 'C'): 78, ('LEU', 'CA'): 79, ('LEU', 'CB'): 80, ('LEU', 'CD1'): 81, ('LEU', 'CD2'): 82, ('LEU', 'CG'): 83, ('LEU', 'N'): 84, ('LEU', 'O'): 85,
    ('LYS', 'C'): 86, ('LYS', 'CA'): 87, ('LYS', 'CB'): 88, ('LYS', 'CD'): 89, ('LYS', 'CE'): 90, ('LYS', 'CG'): 91, ('LYS', 'N'): 92, ('LYS', 'NZ'): 93, ('LYS', 'O'): 94,
    ('MET', 'C'): 95, ('MET', 'CA'): 96, ('MET', 'CB'): 97, ('MET', 'CE'): 98, ('MET', 'CG'): 99, ('MET', 'N'): 100, ('MET', 'O'): 101, ('MET', 'SD'): 102,
    ('PHE', 'C'): 103, ('PHE', 'CA'): 104, ('PHE', 'CB'): 105, ('PHE', 'CD1'): 106, ('PHE', 'CD2'): 107, ('PHE', 'CE1'): 108, ('PHE', 'CE2'): 109, ('PHE', 'CG'): 110, ('PHE', 'CZ'): 111, ('PHE', 'N'): 112, ('PHE', 'O'): 113,
    ('PRO', 'C'): 114, ('PRO', 'CA'): 115, ('PRO', 'CB'): 116, ('PRO', 'CD'): 117, ('PRO', 'CG'): 118, ('PRO', 'N'): 119, ('PRO', 'O'): 120,
    ('SER', 'C'): 121, ('SER', 'CA'): 122, ('SER', 'CB'): 123, ('SER', 'N'): 124, ('SER', 'O'): 125, ('SER', 'OG'): 126,
    ('THR', 'C'): 127, ('THR', 'CA'): 128, ('THR', 'CB'): 129, ('THR', 'CG2'): 130, ('THR', 'N'): 131, ('THR', 'O'): 132, ('THR', 'OG1'): 133,
    ('TRP', 'C'): 134, ('TRP', 'CA'): 135, ('TRP', 'CB'): 136, ('TRP', 'CD1'): 137, ('TRP', 'CD2'): 138, ('TRP', 'CE2'): 139, ('TRP', 'CE3'): 140, ('TRP', 'CG'): 141, ('TRP', 'CH2'): 142, ('TRP', 'CZ2'): 143, ('TRP', 'CZ3'): 144, ('TRP', 'N'): 145, ('TRP', 'NE1'): 146, ('TRP', 'O'): 147,
    ('TYR', 'C'): 148, ('TYR', 'CA'): 149, ('TYR', 'CB'): 150, ('TYR', 'CD1'): 151, ('TYR', 'CD2'): 152, ('TYR', 'CE1'): 153, ('TYR', 'CE2'): 154, ('TYR', 'CG'): 155, ('TYR', 'CZ'): 156, ('TYR', 'N'): 157, ('TYR', 'O'): 158, ('TYR', 'OH'): 159,
    ('VAL', 'C'): 160, ('VAL', 'CA'): 161, ('VAL', 'CB'): 162, ('VAL', 'CG1'): 163, ('VAL', 'CG2'): 164, ('VAL', 'N'): 165, ('VAL', 'O'): 166,
    ('XXX', 'C'): 167, ('XXX', 'N'): 168, ('XXX', 'O'): 169, ('XXX', 'P'): 170, ('XXX', 'S'): 171, ('XXX', 'SE'): 172,
    ('METAL', 'METAL'): 173,
    ('UNK', 'UNK'): 174
}


class AtomFeaturizer:
    """
    Atom-level featurizer for protein structures.
    Extracts atomic features including tokens, coordinates, and SASA.
    """

    def __init__(self):
        """Initialize the atom featurizer."""
        self.res_atm_token = res_atm_token
        self.res_token = res_token
        self.aa_letter = aa_letter

    def get_protein_atom_features(self, pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atom-level features from PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (token, coord):
                - token: torch.Tensor of shape [n_atoms] with atom type tokens
                - coord: torch.Tensor of shape [n_atoms, 3] with 3D coordinates
        """
        token, coord = [], []

        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line[:4] not in ['ATOM', 'HETA'] or (len(line) > 13 and line[13] == 'H'):
                continue

            res_type = line[17:20].strip()
            if res_type == 'HOH':
                continue

            atom_type = line[12:17].strip()

            if atom_type == 'OXT' or res_type in ['LLP', 'PTR']:
                continue
            elif atom_type == res_type[:2]:
                res_type = 'METAL'
                atom_type = 'METAL'
            elif res_type in ['HIS', 'HID', 'HIE', 'HIP']:
                res_type = 'HIS'
            elif res_type in ['CYS', 'CYX', 'CYM']:
                res_type = 'CYS'
            elif res_type not in self.aa_letter:
                res_type = 'XXX'
                if atom_type != 'SE':
                    atom_type = line[13]

            tok = self.res_atm_token.get((res_type, atom_type), 174)  # 174 is UNK token
            xyz = [float(line[i:i+8]) for i in range(30, 54, 8)]

            token.append(tok)
            coord.append(xyz)

        token = torch.tensor(token, dtype=torch.long)
        coord = torch.tensor(coord, dtype=torch.float32)

        return token, coord

    def get_atom_sasa(self, pdb_file: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate atom-level SASA using FreeSASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (atom_sasa, atom_info):
                - atom_sasa: torch.Tensor of shape [n_atoms] with SASA values
                - atom_info: Dictionary containing:
                    - 'residue_name': Residue names for each atom
                    - 'residue_number': Residue numbers
                    - 'atom_name': Atom names
                    - 'chain_label': Chain labels
                    - 'radius': Atomic radii
        """
        # Calculate SASA using FreeSASA
        structure = freesasa.Structure(pdb_file)
        result = freesasa.calc(structure)

        n_atoms = result.nAtoms()

        atom_sasa = []
        residue_names = []
        residue_numbers = []
        atom_names = []
        chain_labels = []
        radii = []

        for i in range(n_atoms):
            # Get SASA value
            sasa = result.atomArea(i)
            atom_sasa.append(sasa)

            # Get atom information
            residue_names.append(structure.residueName(i))
            residue_numbers.append(int(structure.residueNumber(i)))
            atom_names.append(structure.atomName(i).strip())
            chain_labels.append(structure.chainLabel(i))
            radii.append(structure.radius(i))

        # Convert to tensors
        atom_sasa = torch.tensor(atom_sasa, dtype=torch.float32)

        atom_info = {
            'residue_name': residue_names,
            'residue_number': torch.tensor(residue_numbers, dtype=torch.long),
            'atom_name': atom_names,
            'chain_label': chain_labels,
            'radius': torch.tensor(radii, dtype=torch.float32)
        }

        return atom_sasa, atom_info

    def get_all_atom_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get all atom-level features including tokens, coordinates, and SASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary containing:
                - 'token': Atom type tokens [n_atoms]
                - 'coord': 3D coordinates [n_atoms, 3]
                - 'sasa': SASA values [n_atoms]
                - 'residue_token': Residue type for each atom [n_atoms]
                - 'atom_element': Element type for each atom [n_atoms]
                - 'radius': Atomic radii [n_atoms]
        """
        # Get basic atom features
        token, coord = self.get_protein_atom_features(pdb_file)

        # Get SASA features
        atom_sasa, atom_info = self.get_atom_sasa(pdb_file)

        # Extract residue tokens for each atom
        residue_tokens = []
        atom_elements = []

        for res_name in atom_info['residue_name']:
            res_name_clean = res_name.strip()

            # Handle special cases
            if res_name_clean in ['HIS', 'HID', 'HIE', 'HIP']:
                res_name_clean = 'HIS'
            elif res_name_clean in ['CYS', 'CYX', 'CYM']:
                res_name_clean = 'CYS'
            elif res_name_clean not in self.aa_letter:
                if res_name_clean == 'HOH':
                    continue
                res_name_clean = 'XXX'

            res_tok = self.res_token.get(res_name_clean, 20)  # 20 is XXX token
            residue_tokens.append(res_tok)

        # Determine element type from atom token
        for tok in token.tolist():
            # Map token to element based on atom type
            # This is simplified - you might want a more comprehensive mapping
            if tok in range(0, 5):  # ALA atoms
                elements = [2, 2, 2, 3, 4]  # C, C, C, N, O
                atom_elements.append(elements[tok % 5])
            elif tok == 174:  # UNK
                atom_elements.append(0)
            else:
                # Default mapping based on token ranges
                # You can expand this based on the full res_atm_token mapping
                atom_elements.append(2)  # Default to carbon

        # Ensure all tensors have the same length
        min_len = min(len(token), len(atom_sasa))

        features = {
            'token': token[:min_len],
            'coord': coord[:min_len],
            'sasa': atom_sasa[:min_len],
            'residue_token': torch.tensor(residue_tokens[:min_len], dtype=torch.long),
            'atom_element': torch.tensor(atom_elements[:min_len], dtype=torch.long),
            'radius': atom_info['radius'][:min_len] if len(atom_info['radius']) > min_len else atom_info['radius'],
            'metadata': {
                'n_atoms': min_len,
                'residue_names': atom_info['residue_name'][:min_len],
                'residue_numbers': atom_info['residue_number'][:min_len],
                'atom_names': atom_info['atom_name'][:min_len],
                'chain_labels': atom_info['chain_label'][:min_len]
            }
        }

        return features

    def get_residue_aggregated_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get residue-level features by aggregating atom features.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary with residue-aggregated features
        """
        # Get all atom features
        atom_features = self.get_all_atom_features(pdb_file)

        # Group by residue
        residue_numbers = atom_features['metadata']['residue_numbers']
        unique_residues = torch.unique(residue_numbers)

        residue_features = {
            'residue_token': [],
            'center_of_mass': [],
            'total_sasa': [],
            'mean_sasa': [],
            'n_atoms': []
        }

        for res_num in unique_residues:
            mask = residue_numbers == res_num

            # Get residue token (should be same for all atoms in residue)
            res_tokens = atom_features['residue_token'][mask]
            residue_features['residue_token'].append(res_tokens[0])

            # Calculate center of mass
            coords = atom_features['coord'][mask]
            center_of_mass = coords.mean(dim=0)
            residue_features['center_of_mass'].append(center_of_mass)

            # Aggregate SASA
            sasa = atom_features['sasa'][mask]
            residue_features['total_sasa'].append(sasa.sum())
            residue_features['mean_sasa'].append(sasa.mean())

            # Count atoms
            residue_features['n_atoms'].append(mask.sum())

        # Convert to tensors
        for key in residue_features:
            residue_features[key] = torch.stack(residue_features[key]) if key == 'center_of_mass' else torch.tensor(residue_features[key])

        return residue_features


# Convenience function for direct use
def get_protein_atom_features(pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract atom-level features from PDB file.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Tuple of (token, coord)
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_protein_atom_features(pdb_file)


def get_atom_features_with_sasa(pdb_file: str) -> Dict[str, torch.Tensor]:
    """
    Get all atom-level features including SASA.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Dictionary with all atom features
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_all_atom_features(pdb_file)



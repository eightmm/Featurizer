"""
PDB Standardizer Module

This module provides functionality to standardize and clean PDB files for protein analysis.
It handles residue reordering, atom standardization, and removal of unwanted molecules.
"""

import os
from typing import Dict, List, Tuple, Optional


# Standard atoms for each amino acid residue
STANDARD_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'UNK': ['N', 'CA', 'C', 'O', 'CB']
}

# Nucleic acid residues to exclude
NUCLEIC_ACID_RESIDUES = {
    # DNA
    'DA', 'DT', 'DG', 'DC', 'DI', 'DU',
    # RNA
    'A', 'U', 'G', 'C', 'I',
    # Modified nucleotides
    'ADE', 'THY', 'GUA', 'CYT', 'URA',
    '1MA', '2MG', '4SU', '5MC', '5MU', 'PSU', 'H2U', 'M2G', 'OMC', 'OMG'
}


class PDBStandardizer:
    """
    A class for standardizing PDB files.

    This class provides methods to clean and standardize PDB files by:
    - Removing hydrogen atoms (optional)
    - Removing water molecules
    - Removing DNA/RNA residues
    - Reordering atoms according to standard definitions
    - Renumbering residues sequentially
    """

    def __init__(self, remove_hydrogens: bool = True):
        """
        Initialize the PDB standardizer.

        Args:
            remove_hydrogens: Whether to remove hydrogen atoms from the PDB
        """
        self.remove_hydrogens = remove_hydrogens
        self.standard_atoms = STANDARD_ATOMS
        self.nucleic_acid_residues = NUCLEIC_ACID_RESIDUES

    def standardize(self, input_pdb_path: str, output_pdb_path: str) -> str:
        """
        Standardize a PDB file.

        Args:
            input_pdb_path: Path to the input PDB file
            output_pdb_path: Path where the standardized PDB will be saved

        Returns:
            Path to the standardized PDB file
        """
        # Create output directory if needed
        output_dir = os.path.dirname(output_pdb_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Read and parse PDB file
        protein_residues, hetatm_residues = self._parse_pdb(input_pdb_path)

        # Write standardized PDB
        self._write_standardized_pdb(protein_residues, hetatm_residues, output_pdb_path)

        return output_pdb_path

    def _parse_pdb(self, pdb_path: str) -> Tuple[Dict, Dict]:
        """
        Parse a PDB file and extract residue information.

        Args:
            pdb_path: Path to the PDB file

        Returns:
            Tuple of (protein_residues, hetatm_residues) dictionaries
        """
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        protein_residues = {}
        hetatm_residues = {}

        for line in lines:
            if line.startswith('ATOM'):
                self._process_atom_line(line, protein_residues, is_hetatm=False)
            elif line.startswith('HETATM'):
                self._process_atom_line(line, hetatm_residues, is_hetatm=True)

        return protein_residues, hetatm_residues

    def _process_atom_line(self, line: str, residue_dict: Dict, is_hetatm: bool = False):
        """
        Process a single ATOM or HETATM line from PDB file.

        Args:
            line: PDB line to process
            residue_dict: Dictionary to store the residue information
            is_hetatm: Whether this is a HETATM line
        """
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21]
        res_num_str = line[22:27].strip()  # Include insertion code
        element = line[76:78].strip() if len(line) > 76 else atom_name[0]

        # Skip hydrogens if requested
        if self.remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
            return

        # Skip water molecules
        if res_name in ['HOH', 'WAT']:
            return

        # Skip nucleic acid residues
        if res_name in self.nucleic_acid_residues:
            return

        # Store residue information
        residue_key = (chain_id, res_num_str, res_name)
        if residue_key not in residue_dict:
            residue_dict[residue_key] = {}
        residue_dict[residue_key][atom_name] = line

    def _sort_residue_key(self, residue_key: Tuple) -> Tuple:
        """
        Create a sort key for residue ordering.

        Args:
            residue_key: Tuple of (chain_id, res_num_str, res_name)

        Returns:
            Sort key tuple
        """
        chain_id, res_num_str, res_name = residue_key
        # Extract numeric part and insertion code
        res_num = int(''.join(filter(str.isdigit, res_num_str)) or '0')
        insertion_code = ''.join(filter(str.isalpha, res_num_str))
        return (chain_id, res_num, insertion_code)

    def _write_standardized_pdb(self, protein_residues: Dict, hetatm_residues: Dict,
                                output_path: str):
        """
        Write standardized PDB file.

        Args:
            protein_residues: Dictionary of protein residues
            hetatm_residues: Dictionary of HETATM residues
            output_path: Path to write the standardized PDB
        """
        standardized_lines = []
        atom_counter = 1

        # Process protein residues
        standardized_lines, atom_counter = self._write_protein_residues(
            protein_residues, standardized_lines, atom_counter
        )

        # Process HETATM residues
        standardized_lines, atom_counter = self._write_hetatm_residues(
            hetatm_residues, standardized_lines, atom_counter
        )

        # Write to file
        with open(output_path, 'w') as f:
            f.writelines(standardized_lines)

    def _write_protein_residues(self, protein_residues: Dict, lines: List[str],
                                atom_counter: int) -> Tuple[List[str], int]:
        """
        Write protein residues in standardized format.
        """
        # Group by chain
        protein_by_chain = {}
        for residue_key in protein_residues.keys():
            chain_id = residue_key[0]
            if chain_id not in protein_by_chain:
                protein_by_chain[chain_id] = []
            protein_by_chain[chain_id].append(residue_key)

        # Process each chain
        for chain_id in sorted(protein_by_chain.keys()):
            sorted_residues = sorted(protein_by_chain[chain_id], key=self._sort_residue_key)
            res_counter = 1

            for residue_key in sorted_residues:
                chain_id, res_num_str, res_name = residue_key
                residue_atoms = protein_residues[residue_key]

                # Write atoms in standard order if possible
                if res_name in self.standard_atoms:
                    for standard_atom in self.standard_atoms[res_name]:
                        if standard_atom in residue_atoms:
                            line = self._format_atom_line(
                                residue_atoms[standard_atom],
                                atom_counter, res_counter, chain_id, res_name
                            )
                            lines.append(line)
                            atom_counter += 1
                else:
                    # Non-standard residue - write all atoms
                    for atom_name, atom_line in residue_atoms.items():
                        line = self._format_atom_line(
                            atom_line, atom_counter, res_counter, chain_id, res_name
                        )
                        lines.append(line)
                        atom_counter += 1

                res_counter += 1

        return lines, atom_counter

    def _write_hetatm_residues(self, hetatm_residues: Dict, lines: List[str],
                               atom_counter: int) -> Tuple[List[str], int]:
        """
        Write HETATM residues in standardized format.
        """
        # Group by chain
        hetatm_by_chain = {}
        for residue_key in hetatm_residues.keys():
            chain_id = residue_key[0]
            if chain_id not in hetatm_by_chain:
                hetatm_by_chain[chain_id] = []
            hetatm_by_chain[chain_id].append(residue_key)

        # Process each chain
        for chain_id in sorted(hetatm_by_chain.keys()):
            sorted_residues = sorted(hetatm_by_chain[chain_id], key=self._sort_residue_key)
            hetatm_counter = 1

            for residue_key in sorted_residues:
                chain_id, res_num_str, res_name = residue_key
                residue_atoms = hetatm_residues[residue_key]

                for atom_name, atom_line in residue_atoms.items():
                    line = self._format_hetatm_line(
                        atom_line, atom_counter, hetatm_counter, res_name
                    )
                    lines.append(line)
                    atom_counter += 1

                hetatm_counter += 1

        return lines, atom_counter

    def _format_atom_line(self, original_line: str, atom_counter: int,
                         res_counter: int, chain_id: str, res_name: str) -> str:
        """
        Format an ATOM line in standardized PDB format.
        """
        atom_name = original_line[12:16].strip()
        x = float(original_line[30:38])
        y = float(original_line[38:46])
        z = float(original_line[46:54])
        occupancy = original_line[54:60].strip() if len(original_line) > 54 else "1.00"
        temp_factor = original_line[60:66].strip() if len(original_line) > 60 else "0.00"
        element = original_line[76:78].strip() if len(original_line) > 76 else atom_name[0]

        return f"ATOM  {atom_counter:5d}  {atom_name:<4s}{res_name} {chain_id}{res_counter:>4d}    " \
               f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:>6s}{temp_factor:>6s}          {element:>2s}\n"

    def _format_hetatm_line(self, original_line: str, atom_counter: int,
                           hetatm_counter: int, res_name: str) -> str:
        """
        Format a HETATM line in standardized PDB format.
        """
        atom_name = original_line[12:16].strip()
        x = float(original_line[30:38])
        y = float(original_line[38:46])
        z = float(original_line[46:54])
        occupancy = original_line[54:60].strip() if len(original_line) > 54 else "1.00"
        temp_factor = original_line[60:66].strip() if len(original_line) > 60 else "0.00"
        element = original_line[76:78].strip() if len(original_line) > 76 else atom_name[0]

        return f"HETATM{atom_counter:5d} {atom_name:<4s}  {res_name}  {hetatm_counter:>4d}    " \
               f"{x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:>6s}{temp_factor:>6s}           {element:>2s}\n"


def standardize_pdb(input_pdb_path: str, output_pdb_path: str, remove_hydrogens: bool = True) -> str:
    """
    Convenience function to standardize a PDB file.

    Args:
        input_pdb_path: Path to input PDB file
        output_pdb_path: Path for output standardized PDB
        remove_hydrogens: Whether to remove hydrogen atoms

    Returns:
        Path to the standardized PDB file
    """
    standardizer = PDBStandardizer(remove_hydrogens=remove_hydrogens)
    return standardizer.standardize(input_pdb_path, output_pdb_path)



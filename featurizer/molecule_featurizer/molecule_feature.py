"""
Molecule Feature Extractor Module

This module provides functionality to extract features from molecular structures
for machine learning applications. It includes physicochemical descriptors,
fingerprints, and graph-based molecular features in a format similar to ProteinFeaturizer.
"""

import torch
import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional, List
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors, QED, rdPartialCharges
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import rdFingerprintGenerator


class MoleculeFeatureExtractor:
    """
    Extract features from molecules in node/edge format similar to ProteinFeaturizer.
    """

    # Atom type mappings
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'UNK']
    ATOM_TYPE_TO_INT = {atom: i for i, atom in enumerate(ATOM_TYPES)}

    # Bond type mappings
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]

    # Hybridization types
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]

    def __init__(self, use_3d: bool = False):
        """
        Initialize the MoleculeFeatureExtractor.

        Args:
            use_3d: Whether to use 3D coordinates (requires conformer generation)
        """
        self.use_3d = use_3d

    def get_atom_features(self, atom) -> List[float]:
        """
        Get features for a single atom.

        Args:
            atom: RDKit atom object

        Returns:
            List of atom features
        """
        features = []

        # Atom type (one-hot)
        atom_type = atom.GetSymbol()
        if atom_type not in self.ATOM_TYPE_TO_INT:
            atom_type = 'UNK'
        atom_type_idx = self.ATOM_TYPE_TO_INT[atom_type]
        features.extend([1.0 if i == atom_type_idx else 0.0 for i in range(len(self.ATOM_TYPES))])

        # Atomic number (normalized)
        features.append(atom.GetAtomicNum() / 100.0)

        # Degree
        features.append(atom.GetDegree() / 6.0)

        # Formal charge
        features.append(atom.GetFormalCharge() / 5.0)

        # Hybridization (one-hot)
        hybridization = atom.GetHybridization()
        features.extend([1.0 if h == hybridization else 0.0 for h in self.HYBRIDIZATION_TYPES])

        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)

        # Is in ring
        features.append(1.0 if atom.IsInRing() else 0.0)

        # Number of hydrogens
        features.append(atom.GetTotalNumHs() / 4.0)

        # Implicit valence
        features.append(atom.GetImplicitValence() / 6.0)

        # Mass (normalized)
        features.append(atom.GetMass() / 200.0)

        return features

    def get_bond_features(self, bond) -> List[float]:
        """
        Get features for a single bond.

        Args:
            bond: RDKit bond object

        Returns:
            List of bond features
        """
        features = []

        # Bond type (one-hot)
        bond_type = bond.GetBondType()
        features.extend([1.0 if bt == bond_type else 0.0 for bt in self.BOND_TYPES])

        # Is conjugated
        features.append(1.0 if bond.GetIsConjugated() else 0.0)

        # Is in ring
        features.append(1.0 if bond.IsInRing() else 0.0)

        # Stereochemistry
        stereo = bond.GetStereo()
        features.append(1.0 if stereo != Chem.rdchem.BondStereo.STEREONONE else 0.0)

        return features

    def get_physicochemical_features(self, mol) -> np.ndarray:
        """
        Get molecular-level physicochemical descriptors.

        Args:
            mol: RDKit mol object

        Returns:
            Array of physicochemical features
        """
        features = []

        # Molecular weight
        features.append(min(Descriptors.MolWt(mol) / 1000.0, 1.0))

        # LogP
        features.append((Descriptors.MolLogP(mol) + 5) / 10.0)

        # TPSA
        features.append(min(Descriptors.TPSA(mol) / 200.0, 1.0))

        # Rotatable bonds
        features.append(min(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, 1.0))

        # H-bond donors and acceptors
        features.append(min(rdMolDescriptors.CalcNumHBD(mol) / 10.0, 1.0))
        features.append(min(rdMolDescriptors.CalcNumHBA(mol) / 15.0, 1.0))

        # Number of atoms and bonds
        features.append(min(mol.GetNumAtoms() / 100.0, 1.0))
        features.append(min(mol.GetNumBonds() / 120.0, 1.0))

        # Ring counts
        features.append(min(rdMolDescriptors.CalcNumRings(mol) / 10.0, 1.0))
        features.append(min(rdMolDescriptors.CalcNumAromaticRings(mol) / 8.0, 1.0))

        # Heteroatom ratio
        n_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        features.append(n_heteroatoms / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0)

        # QED score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features.append(QED.qed(mol))

        # Lipinski violations
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if rdMolDescriptors.CalcNumHBD(mol) > 5: violations += 1
        if rdMolDescriptors.CalcNumHBA(mol) > 10: violations += 1
        features.append(violations / 4.0)

        # Fraction Csp3
        csp3_count = sum(1 for atom in mol.GetAtoms()
                        if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum() == 6)
        total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        features.append(csp3_count / total_carbons if total_carbons > 0 else 0.0)

        return np.array(features, dtype=np.float32)

    def get_fingerprints(self, mol) -> Dict[str, torch.Tensor]:
        """
        Get molecular fingerprints.

        Args:
            mol: RDKit mol object

        Returns:
            Dictionary of fingerprint tensors
        """
        fingerprints = {}

        # MACCS keys
        fingerprints['maccs'] = torch.tensor(MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=torch.float32)

        # Morgan fingerprint
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            countSimulation=True,
            includeChirality=True
        )
        morgan_fp = morgan_gen.GetFingerprintAsNumPy(mol)
        fingerprints['morgan'] = torch.from_numpy(morgan_fp).float()

        # RDKit fingerprint
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=1,
            maxPath=7,
            fpSize=2048,
            countSimulation=True,
            branchedPaths=True,
            useBondOrder=True
        )
        rdkit_fp = rdkit_gen.GetFingerprintAsNumPy(mol)
        fingerprints['rdkit'] = torch.from_numpy(rdkit_fp).float()

        return fingerprints

    def _extract_node_features(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract node (atom) features from molecule.

        Args:
            mol: RDKit mol object

        Returns:
            Tuple of (node_scalar_features, node_vector_features)
        """
        num_atoms = mol.GetNumAtoms()

        # Collect atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))

        node_scalar_features = torch.tensor(atom_features, dtype=torch.float32)

        # For vector features, we can use partial charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0
            # Handle NaN or infinite values
            if not np.isfinite(float(charge)):
                charge = 0
            charges.append([float(charge)])

        node_vector_features = torch.tensor(charges, dtype=torch.float32)

        return node_scalar_features, node_vector_features

    def _extract_edge_features(self, mol) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Extract edge (bond) features from molecule.

        Args:
            mol: RDKit mol object

        Returns:
            Tuple of (edges, edge_scalar_features, edge_vector_features)
        """
        # Get adjacency information
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
        edge_indices = np.where(adjacency_matrix > 0)

        src_indices = torch.tensor(edge_indices[0], dtype=torch.long)
        dst_indices = torch.tensor(edge_indices[1], dtype=torch.long)
        edges = (src_indices, dst_indices)

        # Collect bond features
        edge_features = []
        edge_vectors = []

        for i, j in zip(edge_indices[0], edge_indices[1]):
            if i < j:  # Avoid duplicate bonds
                bond = mol.GetBondBetweenAtoms(int(i), int(j))
                if bond is not None:
                    # Add features for both directions
                    bond_feat = self.get_bond_features(bond)
                    edge_features.append(bond_feat)
                    edge_features.append(bond_feat)  # Same features for reverse direction

                    # Simple edge vector (could be enhanced with 3D coordinates)
                    edge_vectors.append([1.0, 0.0, 0.0])  # Placeholder
                    edge_vectors.append([-1.0, 0.0, 0.0])  # Reverse direction

        if edge_features:
            edge_scalar_features = torch.tensor(edge_features, dtype=torch.float32)
            edge_vector_features = torch.tensor(edge_vectors, dtype=torch.float32)
        else:
            # Handle molecules with no bonds
            edge_scalar_features = torch.zeros((0, 7), dtype=torch.float32)
            edge_vector_features = torch.zeros((0, 3), dtype=torch.float32)

        return edges, edge_scalar_features, edge_vector_features

    def get_coordinates(self, mol) -> Optional[torch.Tensor]:
        """
        Get 3D coordinates for atoms (if available or generated).

        Args:
            mol: RDKit mol object

        Returns:
            Tensor of coordinates or None
        """
        if self.use_3d:
            # Check if mol has conformer
            if mol.GetNumConformers() == 0:
                # Generate 3D conformer
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol)

            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                return torch.tensor(coords, dtype=torch.float32)

        # Return 2D placeholder coordinates
        num_atoms = mol.GetNumAtoms()
        return torch.zeros((num_atoms, 3), dtype=torch.float32)

    def get_features(self, mol_or_smiles) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract all features for the molecule in node/edge format.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string

        Returns:
            Tuple of (node_features, edge_features) dictionaries
        """
        # Parse molecule
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            mol = mol_or_smiles

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Get coordinates
        coords = self.get_coordinates(mol)

        # Extract node features
        node_scalar_features, node_vector_features = self._extract_node_features(mol)

        # Extract edge features
        edges, edge_scalar_features, edge_vector_features = self._extract_edge_features(mol)

        # Get molecular-level features
        mol_descriptors = self.get_physicochemical_features(mol)

        # Package node features
        node = {
            'coord': coords,  # [num_atoms, 3]
            'node_scalar_features': (node_scalar_features,),  # Tuple format like ProteinFeaturizer
            'node_vector_features': (node_vector_features,),  # Tuple format
            'mol_descriptors': mol_descriptors  # Additional molecular-level features
        }

        # Package edge features
        edge = {
            'edges': edges,  # (src_indices, dst_indices)
            'edge_scalar_features': (edge_scalar_features,),  # Tuple format
            'edge_vector_features': (edge_vector_features,)  # Tuple format
        }

        return node, edge

    def extract_all_features(self, mol, add_hs=True):
        """
        Legacy method for backward compatibility.
        Extracts all molecular features including fingerprints.

        Args:
            mol: RDKit mol object
            add_hs: Whether to add hydrogens

        Returns:
            Dictionary containing descriptors and fingerprints
        """
        if add_hs:
            mol = Chem.AddHs(mol)

        # Get node/edge features
        node, edge = self.get_features(mol)

        # Get fingerprints
        fingerprints = self.get_fingerprints(mol)

        # Combine into legacy format
        result = {
            'descriptor': torch.from_numpy(node['mol_descriptors']),
            **fingerprints
        }

        return result


def create_molecule_features(mol_or_smiles, add_hs=True):
    """
    Create molecule features from RDKit mol object or SMILES string.

    Args:
        mol_or_smiles: RDKit mol object or SMILES string
        add_hs: Whether to add hydrogens (default: True)

    Returns:
        Dictionary containing molecule features
    """
    extractor = MoleculeFeatureExtractor()

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    return extractor.extract_all_features(mol, add_hs=add_hs)


if __name__ == "__main__":
    # Example usage
    smiles = "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid

    # Create extractor
    extractor = MoleculeFeatureExtractor()

    # Get features in node/edge format (like ProteinFeaturizer)
    node, edge = extractor.get_features(smiles)

    print(f"Molecule: {smiles}")
    print(f"\nNode features:")
    print(f"  - Coordinates shape: {node['coord'].shape}")
    print(f"  - Node scalar features: {node['node_scalar_features'][0].shape}")
    print(f"  - Node vector features: {node['node_vector_features'][0].shape}")
    print(f"  - Molecular descriptors: {node['mol_descriptors'].shape}")

    print(f"\nEdge features:")
    print(f"  - Number of edges: {len(edge['edges'][0])}")
    if len(edge['edge_scalar_features'][0]) > 0:
        print(f"  - Edge scalar features: {edge['edge_scalar_features'][0].shape}")
        print(f"  - Edge vector features: {edge['edge_vector_features'][0].shape}")

    # Legacy format
    features = create_molecule_features(smiles)
    print(f"\nLegacy format features:")
    print(f"  - Descriptors: {features['descriptor'].shape}")
    print(f"  - Morgan FP: {features['morgan'].shape}")
    print(f"  - MACCS keys: {features['maccs'].shape}")
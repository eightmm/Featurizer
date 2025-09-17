import torch
import dgl
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors, QED, rdPartialCharges, AllChem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import rdFingerprintGenerator
from typing import Union, Dict, Optional, Tuple


class MoleculeFeaturizer:
    """
    Unified molecule featurizer for extracting molecular features and graph representations.

    This class provides methods to extract both molecular-level features (descriptors and fingerprints)
    and graph-level features (node and edge features) from RDKit mol objects or SMILES strings.
    """

    ATOMS = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'UNK']
    PERIODS = list(range(5))
    GROUPS = list(range(18))
    DEGREES = list(range(7))
    HEAVY_DEGREES = list(range(7))
    VALENCES = list(range(8))
    TOTAL_HS = list(range(5))
    HYBRIDS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]
    BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    STEREOS = [Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOCIS,
               Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOTRANS, Chem.rdchem.BondStereo.STEREOZ]

    PERIODIC = {'H': (0, 0), 'He': (0, 17), 'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13),
                'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17), 'Na': (2, 0), 'Mg': (2, 1),
                'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
                'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5),
                'Mn': (3, 6), 'Fe': (3, 7), 'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11),
                'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
                'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5),
                'Tc': (4, 6), 'Ru': (4, 7), 'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11),
                'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17)}

    ELECTRONEGATIVITY = {(0, 0): 2.20, (1, 0): 0.98, (1, 1): 1.57, (1, 12): 2.04, (1, 13): 2.55,
                        (1, 14): 3.04, (1, 15): 3.44, (1, 16): 3.98, (2, 0): 0.93, (2, 1): 1.31,
                        (2, 12): 1.61, (2, 13): 1.90, (2, 14): 2.19, (2, 15): 2.58, (2, 16): 3.16,
                        (3, 0): 0.82, (3, 1): 1.00, (3, 2): 1.36, (3, 3): 1.54, (3, 4): 1.63,
                        (3, 5): 1.66, (3, 6): 1.55, (3, 7): 1.83, (3, 8): 1.88, (3, 9): 1.91,
                        (3, 10): 1.90, (3, 11): 1.65, (3, 12): 1.81, (3, 13): 2.01, (3, 14): 2.18,
                        (3, 15): 2.55, (3, 16): 2.96, (3, 17): 3.00, (4, 0): 0.82, (4, 1): 0.95,
                        (4, 2): 1.22, (4, 3): 1.33, (4, 4): 1.60, (4, 5): 2.16, (4, 6): 1.90,
                        (4, 7): 2.20, (4, 8): 2.28, (4, 9): 2.20, (4, 10): 1.93, (4, 11): 1.69,
                        (4, 12): 1.78, (4, 13): 1.96, (4, 14): 2.05, (4, 15): 2.10, (4, 16): 2.66, (4, 17): 2.60}

    BASIC_SMARTS = {
        "h_accept": "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
        "h_donor": "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
        "e_accept": "[$([C,S](=[O,S,P])-[O;H1,-1])]",
        "e_donor": "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]",
        "hydrophobic": "[C,c,S&H0&v2,F,Cl,Br,I&!$(C=[O,N,P,S])&!$(C#N);!$(C=O)]"
    }


    def __init__(self):
        self.rotate_smarts = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"

    @staticmethod
    def one_hot(x, allowable_set):
        return [x == s for s in (allowable_set if x in allowable_set else allowable_set[:-1] + [allowable_set[-1]])]

    def _prepare_mol(self, mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True) -> Chem.Mol:
        """
        Prepare molecule from SMILES string or RDKit mol object.
        Preserves 3D coordinates when adding hydrogens if the molecule has a conformer.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            RDKit mol object with optional hydrogens
        """
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            mol = mol_or_smiles

        if add_hs and mol is not None:
            # Check if molecule has 3D coordinates
            has_3d_coords = mol.GetNumConformers() > 0
            if has_3d_coords:
                # Preserve 3D coordinates when adding hydrogens
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                # No 3D coords, just add hydrogens
                mol = Chem.AddHs(mol)

        return mol

    def get_physicochemical_features(self, mol):
        """Extract physicochemical features from molecule."""
        features = {}

        features['mw'] = min(Descriptors.MolWt(mol) / 1000.0, 1.0)
        features['logp'] = (Descriptors.MolLogP(mol) + 5) / 10.0
        features['tpsa'] = min(Descriptors.TPSA(mol) / 200.0, 1.0)

        features['n_rotatable_bonds'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, 1.0)
        features['flexibility'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0, 1.0)

        features['hbd'] = min(rdMolDescriptors.CalcNumHBD(mol) / 10.0, 1.0)
        features['hba'] = min(rdMolDescriptors.CalcNumHBA(mol) / 15.0, 1.0)

        features['n_atoms'] = min(mol.GetNumAtoms() / 100.0, 1.0)
        features['n_bonds'] = min(mol.GetNumBonds() / 120.0, 1.0)
        features['n_rings'] = min(rdMolDescriptors.CalcNumRings(mol) / 10.0, 1.0)
        features['n_aromatic_rings'] = min(rdMolDescriptors.CalcNumAromaticRings(mol) / 8.0, 1.0)

        n_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        features['heteroatom_ratio'] = n_heteroatoms / mol.GetNumAtoms()

        return features

    def get_druglike_features(self, mol):
        """Extract drug-likeness features from molecule."""
        features = {}

        violations = 0
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1

        features['lipinski_violations'] = violations / 4.0
        features['passes_lipinski'] = 1.0 if violations == 0 else 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features['qed'] = QED.qed(mol)

        features['num_heavy_atoms'] = min(mol.GetNumHeavyAtoms() / 50.0, 1.0)

        csp3_count = sum(1 for atom in mol.GetAtoms()
                        if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum() == 6)
        total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        features['frac_csp3'] = csp3_count / total_carbons if total_carbons > 0 else 0.0

        return features


    def get_structural_features(self, mol):
        """Extract structural features from molecule."""
        features = {}

        ring_info = mol.GetRingInfo()
        features['n_ring_systems'] = min(len(ring_info.AtomRings()) / 8.0, 1.0)

        max_ring_size = max([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0
        features['max_ring_size'] = min(max_ring_size / 12.0, 1.0)

        avg_ring_size = np.mean([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0
        features['avg_ring_size'] = min(avg_ring_size / 8.0, 1.0)

        return features

    def get_fingerprints(self, mol):
        """Extract various molecular fingerprints."""
        fingerprints = {}

        fingerprints['maccs'] = torch.tensor(MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=torch.float32)

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            countSimulation=True,
            includeChirality=True
        )

        morgan_fp = morgan_gen.GetFingerprintAsNumPy(mol)
        morgan_count_fp = morgan_gen.GetCountFingerprintAsNumPy(mol)
        fingerprints['morgan'] = torch.from_numpy(morgan_fp).float()
        fingerprints['morgan_count'] = torch.from_numpy(morgan_count_fp).float()

        feature_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            countSimulation=True
        )
        feature_morgan_fp = feature_morgan_gen.GetFingerprintAsNumPy(mol)
        fingerprints['feature_morgan'] = torch.from_numpy(feature_morgan_fp).float()

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

        ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=1,
            maxDistance=8,
            fpSize=2048,
            countSimulation=True
        )
        ap_fp = ap_gen.GetFingerprintAsNumPy(mol)
        fingerprints['atom_pair'] = torch.from_numpy(ap_fp).float()

        tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=4,
            fpSize=2048,
            countSimulation=True
        )
        tt_fp = tt_gen.GetFingerprintAsNumPy(mol)
        fingerprints['topological_torsion'] = torch.from_numpy(tt_fp).float()

        pharm_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        bit_vector = torch.zeros(1024)
        for bit_id in pharm_fp.GetOnBits():
            if bit_id < 1024:
                bit_vector[bit_id] = 1.0
        fingerprints['pharmacophore2d'] = bit_vector.float()

        return fingerprints

    def get_feature(self, mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True) -> Dict:
        """
        Extract all molecular-level features including descriptors and fingerprints.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            Dictionary containing descriptor tensor and fingerprint tensors
        """
        mol = self._prepare_mol(mol_or_smiles, add_hs)

        physicochemical_features = self.get_physicochemical_features(mol)
        druglike_features = self.get_druglike_features(mol)
        structural_features = self.get_structural_features(mol)

        all_descriptors = []

        descriptor_keys = [
            'mw', 'logp', 'tpsa', 'n_rotatable_bonds', 'flexibility',
            'hbd', 'hba', 'n_atoms', 'n_bonds', 'n_rings', 'n_aromatic_rings',
            'heteroatom_ratio'
        ]
        for key in descriptor_keys:
            all_descriptors.append(float(physicochemical_features[key]))

        druglike_keys = [
            'lipinski_violations', 'passes_lipinski', 'qed',
            'num_heavy_atoms', 'frac_csp3'
        ]
        for key in druglike_keys:
            all_descriptors.append(float(druglike_features[key]))

        structural_keys = ['n_ring_systems', 'max_ring_size', 'avg_ring_size']
        for key in structural_keys:
            all_descriptors.append(float(structural_features[key]))

        descriptor_tensor = torch.tensor(all_descriptors, dtype=torch.float32)

        fingerprints = self.get_fingerprints(mol)

        result = {
            'descriptor': descriptor_tensor,
            'maccs': fingerprints['maccs'],
            'morgan': fingerprints['morgan'],
            'morgan_count': fingerprints['morgan_count'],
            'feature_morgan': fingerprints['feature_morgan'],
            'rdkit': fingerprints['rdkit'],
            'atom_pair': fingerprints['atom_pair'],
            'topological_torsion': fingerprints['topological_torsion'],
            'pharmacophore2d': fingerprints['pharmacophore2d']
        }

        return result

    def get_ring_mappings(self, mol):
        """Get ring mapping information for atoms and bonds."""
        ring_info = mol.GetRingInfo()
        atom_rings = {i: [] for i in range(mol.GetNumAtoms())}
        bond_rings = {i: [] for i in range(mol.GetNumBonds())}

        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                atom_rings[atom_idx].append(len(ring))

        for ring in ring_info.BondRings():
            for bond_idx in ring:
                bond_rings[bond_idx].append(len(ring))

        return atom_rings, bond_rings

    def get_degree_features(self, mol):
        """Get degree-related features for atoms."""
        degree_info = {}

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            neighbors = [n for n in atom.GetNeighbors()]

            total_degree = atom.GetDegree()
            heavy_degree = len([n for n in neighbors if n.GetAtomicNum() > 1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()

            neighbor_degrees = [n.GetDegree() for n in neighbors]
            neighbor_heavy_degrees = [len([nn for nn in n.GetNeighbors() if nn.GetAtomicNum() > 1]) for n in neighbors]

            if neighbor_degrees:
                min_neighbor_deg = min(neighbor_degrees)
                max_neighbor_deg = max(neighbor_degrees)
                mean_neighbor_deg = sum(neighbor_degrees) / len(neighbor_degrees)
                min_neighbor_heavy = min(neighbor_heavy_degrees)
                max_neighbor_heavy = max(neighbor_heavy_degrees)
                mean_neighbor_heavy = sum(neighbor_heavy_degrees) / len(neighbor_heavy_degrees)
            else:
                min_neighbor_deg = max_neighbor_deg = mean_neighbor_deg = 0
                min_neighbor_heavy = max_neighbor_heavy = mean_neighbor_heavy = 0

            degree_centrality = total_degree / (mol.GetNumAtoms() - 1) if mol.GetNumAtoms() > 1 else 0

            degree_info[atom_idx] = {
                'total_degree': total_degree,
                'heavy_degree': heavy_degree,
                'valence': total_valence,
                'min_neighbor_deg': min_neighbor_deg,
                'max_neighbor_deg': max_neighbor_deg,
                'mean_neighbor_deg': mean_neighbor_deg,
                'min_neighbor_heavy': min_neighbor_heavy,
                'max_neighbor_heavy': max_neighbor_heavy,
                'mean_neighbor_heavy': mean_neighbor_heavy,
                'degree_centrality': degree_centrality,
                'degree_variance': sum([(d - mean_neighbor_deg)**2 for d in neighbor_degrees]) / len(neighbor_degrees) if neighbor_degrees else 0
            }

        return degree_info


    def get_stereochemistry_features(self, mol):
        """Get stereochemistry features for atoms."""
        num_atoms = mol.GetNumAtoms()
        stereo_features = torch.zeros(num_atoms, 8)

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()

            chiral_tag = atom.GetChiralTag()
            if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                stereo_features[atom_idx, 0] = 1.0
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                stereo_features[atom_idx, 1] = 1.0
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                stereo_features[atom_idx, 2] = 1.0

            if (len(atom.GetNeighbors()) == 4 and
                atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3):
                stereo_features[atom_idx, 3] = 1.0

            for bond in atom.GetBonds():
                if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                    stereo_features[atom_idx, 4] = 1.0
                    break

            if atom.GetIsAromatic():
                stereo_features[atom_idx, 5] = 1.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                stereo_features[atom_idx, 6] = 1.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                stereo_features[atom_idx, 7] = 1.0

        return stereo_features

    def get_partial_charges(self, mol):
        """Get partial charges for atoms."""
        num_atoms = mol.GetNumAtoms()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdPartialCharges.ComputeGasteigerCharges(mol)

        charges = torch.zeros(num_atoms, 2)

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            charge = float(atom.GetProp('_GasteigerCharge'))

            if torch.isnan(torch.tensor(charge)):
                charge = 0.0

            charge = max(-1.0, min(1.0, charge))
            charges[atom_idx, 0] = (charge + 1.0) / 2.0
            charges[atom_idx, 1] = abs(charge)

        return charges

    def get_extended_neighborhood(self, mol):
        """Get extended neighborhood features."""
        num_atoms = mol.GetNumAtoms()
        ext_features = torch.zeros(num_atoms, 6)

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()

            neighbors_1 = list(atom.GetNeighbors())

            neighbors_2 = set()
            for n1 in neighbors_1:
                for n2 in n1.GetNeighbors():
                    if n2.GetIdx() != atom_idx:
                        neighbors_2.add(n2)
            neighbors_2 = list(neighbors_2)

            if neighbors_1:
                ext_features[atom_idx, 0] = min(len(neighbors_1) / 6.0, 1.0)
                ext_features[atom_idx, 1] = sum(n.GetIsAromatic() for n in neighbors_1) / len(neighbors_1)
                ext_features[atom_idx, 2] = sum(1 for n in neighbors_1 if n.GetSymbol() in ['N', 'O', 'S']) / len(neighbors_1)

            if neighbors_2:
                ext_features[atom_idx, 3] = min(len(neighbors_2) / 20.0, 1.0)
                ext_features[atom_idx, 4] = sum(n.GetIsAromatic() for n in neighbors_2) / len(neighbors_2)
                ext_features[atom_idx, 5] = sum(1 for n in neighbors_2 if n.GetSymbol() in ['N', 'O', 'S']) / len(neighbors_2)

        return ext_features


    def get_ring_features(self, sizes, is_aromatic):
        """Get ring-related features."""
        is_in_ring = len(sizes) > 0
        num_rings = min(len(sizes), 4)
        smallest = min(sizes) if sizes else 0

        size_features = [False] * 6
        for size in sizes:
            if 3 <= size <= 8:
                size_features[size - 3] = True
            elif size > 8:
                size_features[5] = True

        return ([is_in_ring, is_aromatic, num_rings] +
                size_features +
                self.one_hot(num_rings, [0, 1, 2, 3, 4]) +
                self.one_hot(smallest, [0, 3, 4, 5, 6, 7, 8]))

    def get_3d_coordinates(self, mol):
        """
        Get 3D coordinates for atoms if available, otherwise generate them.
        If the molecule already has conformers, use the first one.
        Otherwise, generate 3D coordinates using ETKDG method.

        Args:
            mol: RDKit mol object

        Returns:
            Tensor of shape (n_atoms, 3) containing 3D coordinates
        """
        # Check if molecule already has conformers
        if mol.GetNumConformers() > 0:
            # Use existing conformer
            conf = mol.GetConformer(0)
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return torch.tensor(coords, dtype=torch.float32)
        else:
            # Generate 3D coordinates
            try:
                # Use ETKDG method for better 3D structure generation
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                if mol.GetNumConformers() > 0:
                    # Optimize the geometry
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                    conf = mol.GetConformer(0)
                    coords = []
                    for i in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(i)
                        coords.append([pos.x, pos.y, pos.z])
                    return torch.tensor(coords, dtype=torch.float32)
                else:
                    # If embedding fails, return zero coordinates
                    return torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float32)
            except:
                # Fallback to zero coordinates if any error occurs
                return torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float32)

    def get_atom_features(self, mol):
        """Get comprehensive atom features including 3D coordinates if available."""
        atom_rings, _ = self.get_ring_mappings(mol)
        degree_info = self.get_degree_features(mol)

        stereo_feat = self.get_stereochemistry_features(mol)
        charge_feat = self.get_partial_charges(mol)
        ext_neighbor_feat = self.get_extended_neighborhood(mol)

        features = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            period, group = self.PERIODIC.get(symbol, (5, 18))
            electronegativity = self.ELECTRONEGATIVITY.get((period, group), 0.0)
            deg_info = degree_info[atom_idx]

            basic_feat = (
                self.one_hot(symbol, self.ATOMS) +
                self.one_hot(period, self.PERIODS) +
                self.one_hot(group, self.GROUPS) +
                [atom.GetIsAromatic(), atom.IsInRing(),
                 min(atom.GetNumRadicalElectrons() / 3.0, 1.0),
                 (atom.GetFormalCharge() + 3) / 6.0,
                 (electronegativity - 0.8) / 3.2]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()
                total_hs = atom.GetTotalNumHs()

            degree_feat = (
                self.one_hot(deg_info['total_degree'], self.DEGREES) +
                self.one_hot(deg_info['heavy_degree'], self.HEAVY_DEGREES) +
                self.one_hot(total_valence, self.VALENCES) +
                self.one_hot(total_hs, self.TOTAL_HS) +
                self.one_hot(atom.GetHybridization(), self.HYBRIDS) +
                [deg_info['min_neighbor_deg'] / 6, deg_info['max_neighbor_deg'] / 6,
                 deg_info['mean_neighbor_deg'] / 6, deg_info['min_neighbor_heavy'] / 6,
                 deg_info['max_neighbor_heavy'] / 6, deg_info['mean_neighbor_heavy'] / 6,
                 deg_info['degree_centrality'], deg_info['degree_variance'] / 10]
            )

            ring_feat = self.get_ring_features(atom_rings[atom_idx], atom.GetIsAromatic())
            features.append(basic_feat + degree_feat + ring_feat)

        basic_smarts_feat = torch.zeros(mol.GetNumAtoms(), len(self.BASIC_SMARTS))
        for idx, smarts in enumerate(self.BASIC_SMARTS.values()):
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if matches:
                basic_smarts_feat[list(sum(matches, ())), idx] = 1

        atom_feat = torch.tensor(features, dtype=torch.float32)
        node_features = torch.cat([atom_feat, basic_smarts_feat,
                                   stereo_feat, charge_feat, ext_neighbor_feat], dim=-1)

        coords = self.get_3d_coordinates(mol)

        return node_features, coords

    def get_bond_features(self, mol):
        """Get comprehensive bond features."""
        _, bond_rings = self.get_ring_mappings(mol)
        degree_info = self.get_degree_features(mol)
        rotatable_bonds = set(sum(mol.GetSubstructMatches(Chem.MolFromSmarts(self.rotate_smarts)), ()))

        adj = torch.zeros(mol.GetNumAtoms(), mol.GetNumAtoms(), 44)
        bond_indices = torch.triu(torch.tensor(Chem.GetAdjacencyMatrix(mol))).nonzero()

        for src, dst in bond_indices:
            bond = mol.GetBondBetweenAtoms(src.item(), dst.item())
            src_deg = degree_info[src.item()]
            dst_deg = degree_info[dst.item()]

            basic_feat = (
                self.one_hot(bond.GetBondType(), self.BONDS) +
                self.one_hot(bond.GetStereo(), self.STEREOS) +
                [bond.IsInRing(), bond.GetIsConjugated(), (src, dst) in rotatable_bonds]
            )

            bond_degree_feat = [
                abs(src_deg['total_degree'] - dst_deg['total_degree']) / 6,
                abs(src_deg['heavy_degree'] - dst_deg['heavy_degree']) / 6,
                abs(src_deg['valence'] - dst_deg['valence']) / 8,
                (src_deg['total_degree'] + dst_deg['total_degree']) / 12,
                (src_deg['heavy_degree'] + dst_deg['heavy_degree']) / 12,
                (src_deg['valence'] + dst_deg['valence']) / 16,
                abs(src_deg['degree_centrality'] - dst_deg['degree_centrality']),
                (src_deg['degree_centrality'] + dst_deg['degree_centrality']) / 2,
                min(src_deg['total_degree'], dst_deg['total_degree']) / 6,
                max(src_deg['total_degree'], dst_deg['total_degree']) / 6
            ]

            ring_feat = self.get_ring_features(bond_rings[bond.GetIdx()], bond.GetIsAromatic())

            adj[src, dst] = torch.tensor(basic_feat + bond_degree_feat + ring_feat, dtype=torch.float32)

        return adj + adj.transpose(0, 1)

    def get_graph(self, mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True) -> Dict:
        """
        Create molecular graph with node and edge features from molecule.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            Dictionary with 'node' and 'edge' keys containing feature tensors,
            similar to protein featurizer format
        """
        mol = self._prepare_mol(mol_or_smiles, add_hs)

        node_features, coords = self.get_atom_features(mol)
        bond_features = self.get_bond_features(mol)

        src, dst = torch.where(bond_features.sum(dim=-1) > 0)
        edge_features = bond_features[src, dst]

        result = {
            'node': {
                'features': node_features,
                'coords': coords,
                'num_nodes': mol.GetNumAtoms()
            },
            'edge': {
                'src': src,
                'dst': dst,
                'features': edge_features,
                'num_edges': len(src)
            }
        }

        return result
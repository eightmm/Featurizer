import torch
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors, QED
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from rdkit.Chem import rdFingerprintGenerator

class MolecularFeatureExtractor:
    """Extract universal molecular features from RDKit mol objects."""

        
    def get_physicochemical_features(self, mol):
        features = {}
        
        features['mw'] = min(Descriptors.MolWt(mol) / 1000.0, 1.0)  # 분자량 (최대 1000으로 제한)
        features['logp'] = (Descriptors.MolLogP(mol) + 5) / 10.0  # LogP (-5~5 -> 0~1)
        features['tpsa'] = min(Descriptors.TPSA(mol) / 200.0, 1.0)  # TPSA (최대 200으로 제한)
        
        features['n_rotatable_bonds'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, 1.0)
        features['flexibility'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0, 1.0)
        
        features['hbd'] = min(rdMolDescriptors.CalcNumHBD(mol) / 10.0, 1.0)  # H-bond donors
        features['hba'] = min(rdMolDescriptors.CalcNumHBA(mol) / 15.0, 1.0)  # H-bond acceptors
        
        features['n_atoms'] = min(mol.GetNumAtoms() / 100.0, 1.0)
        features['n_bonds'] = min(mol.GetNumBonds() / 120.0, 1.0)
        features['n_rings'] = min(rdMolDescriptors.CalcNumRings(mol) / 10.0, 1.0)
        features['n_aromatic_rings'] = min(rdMolDescriptors.CalcNumAromaticRings(mol) / 8.0, 1.0)
        
        n_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        features['heteroatom_ratio'] = n_heteroatoms / mol.GetNumAtoms()
        
        return features

    def get_druglike_features(self, mol):
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

    def get_atom_composition_features(self, mol):
        """Get universal atom composition features."""
        features = {}

        n_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
        n_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        n_sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
        n_halogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53])
        n_phosphorus = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)

        total_atoms = mol.GetNumAtoms()
        features['nitrogen_ratio'] = n_nitrogen / total_atoms if total_atoms > 0 else 0.0
        features['oxygen_ratio'] = n_oxygen / total_atoms if total_atoms > 0 else 0.0
        features['sulfur_ratio'] = n_sulfur / total_atoms if total_atoms > 0 else 0.0
        features['halogen_ratio'] = n_halogens / total_atoms if total_atoms > 0 else 0.0
        features['phosphorus_ratio'] = n_phosphorus / total_atoms if total_atoms > 0 else 0.0

        return features

    def get_structural_features(self, mol):
        features = {}
        
        ring_info = mol.GetRingInfo()
        features['n_ring_systems'] = min(len(ring_info.AtomRings()) / 8.0, 1.0)
        
        max_ring_size = max([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0
        features['max_ring_size'] = min(max_ring_size / 12.0, 1.0)
        
        avg_ring_size = np.mean([len(ring) for ring in ring_info.AtomRings()]) if ring_info.AtomRings() else 0
        features['avg_ring_size'] = min(avg_ring_size / 8.0, 1.0)
        
        return features

    def get_fingerprints(self, mol):
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
    
    def extract_all_features(self, mol, add_hs=True):
        """Extract all molecular features from an RDKit mol object.

        Args:
            mol: RDKit mol object
            add_hs: Whether to add hydrogens (default: True)

        Returns:
            Dictionary containing descriptors and fingerprints
        """
        if add_hs:
            mol = Chem.AddHs(mol)
        
        # Get all individual feature dictionaries
        physicochemical_features = self.get_physicochemical_features(mol)
        druglike_features = self.get_druglike_features(mol)
        atom_composition_features = self.get_atom_composition_features(mol)
        structural_features = self.get_structural_features(mol)
        
        # Combine all descriptors into a single list
        all_descriptors = []
        
        # Add physicochemical features in a consistent order
        descriptor_keys = [
            'mw', 'logp', 'tpsa', 'n_rotatable_bonds', 'flexibility',
            'hbd', 'hba', 'n_atoms', 'n_bonds', 'n_rings', 'n_aromatic_rings',
            'heteroatom_ratio'
        ]
        for key in descriptor_keys:
            all_descriptors.append(float(physicochemical_features[key]))
        
        # Add druglike features
        druglike_keys = [
            'lipinski_violations', 'passes_lipinski', 'qed', 
            'num_heavy_atoms', 'frac_csp3'
        ]
        for key in druglike_keys:
            all_descriptors.append(float(druglike_features[key]))
        
        # Add atom composition features
        atom_comp_keys = [
            'nitrogen_ratio', 'oxygen_ratio', 'sulfur_ratio', 'halogen_ratio', 'phosphorus_ratio'
        ]
        for key in atom_comp_keys:
            all_descriptors.append(float(atom_composition_features[key]))
        
        # Add structural features
        structural_keys = ['n_ring_systems', 'max_ring_size', 'avg_ring_size']
        for key in structural_keys:
            all_descriptors.append(float(structural_features[key]))
        
        # Convert to torch tensor
        descriptor_tensor = torch.tensor(all_descriptors, dtype=torch.float32)
        
        # Get fingerprints
        fingerprints = self.get_fingerprints(mol)
        
        # Return in the requested format: descriptor, fp1, fp2, ...
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


def create_molecular_features(mol_or_smiles, add_hs=True):
    """Create molecular features from RDKit mol object or SMILES string.

    Args:
        mol_or_smiles: RDKit mol object or SMILES string
        add_hs: Whether to add hydrogens (default: True)

    Returns:
        Dictionary containing molecular features
    """
    extractor = MolecularFeatureExtractor()

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    return extractor.extract_all_features(mol, add_hs=add_hs)

if __name__ == "__main__":
    # Example with SMILES
    smiles = "C1=CC=C(C=C1)C(=O)O"
    features = create_molecular_features(smiles)
    print("Features from SMILES:", features['descriptor'].shape)

    # Example with mol object
    mol = Chem.MolFromSmiles(smiles)
    features2 = create_molecular_features(mol)
    print("Features from mol object:", features2['descriptor'].shape)
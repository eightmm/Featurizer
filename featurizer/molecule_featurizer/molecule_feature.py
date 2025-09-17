import torch
import dgl
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors, QED, rdPartialCharges
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import rdFingerprintGenerator


class MoleculeFeaturizer:
    CYP3A4_INHIBITOR_SMARTS = {
        "imidazole": "c1ncn[c,n]1",
        "triazole": "c1nn[c,n]n1",
        "azole_general": "[n;r5;a]c[n;r5;a]",
        "pyridine": "[n;r6;a]",
        "furan": "o1cccc1",
        "tertiary_amine": "[NX3;!$(N=O);!$(N-C=O)]([CH3])([CH3])",
        "terminal_acetylene": "[C]#[CH1]",
        "quinoline": "c1cccc2ncccc12",
        "benzimidazole": "c1nc2ccccc2[nH]1",
        "macrolide_amine": "[NX3]([CH3])([CH3])[CH2]",
    }
    
    CYP3A4_SUBSTRATE_SMARTS = {
        "n_dealkylation": "[NX3,NX4+;!$(N-C=O)]([CH3,CH2CH3])",
        "o_dealkylation_aromatic": "[O;X2;!$(O-C=O)]([CH3,CH2CH3])c",
        "o_dealkylation_aliphatic": "[O;X2;!$(O-C=O)]([CH3,CH2CH3])C",
        "aromatic_hydroxylation": "[c;H1]",
        "benzylic_hydroxylation": "[C;H2,H3][c]",
        "allylic_hydroxylation": "[C;H2,H3][C]=[C]",
        "aliphatic_hydroxylation": "[C;H2,H3][C;H2][C;H2]",
        "epoxidation": "[C]=[C]",
        "s_oxidation": "[SX2]",
        "n_oxidation": "[NX3;!$(N=O);!$(N-[S,P]=O)]",
        "deamination": "[C][CH2][NH2]",
        "ester_hydrolysis": "[O][C](=O)[C]",
        "amide_hydrolysis": "[N][C](=O)[C]",
        "cyp3a4_hotspot": "[c]1[c][c][c]([O,N,S])[c][c]1",
        "steroid_scaffold": "[C][C][C][C]1[C][C][C]2[C][C][C][C][C]12",
    }

    # Graph building constants
    ATOMS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'UNK']
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

    # 패턴별 신뢰도 가중치 (0.1: 약한신호, 0.5: 중간, 1.0: 강한신호)
    INHIBITOR_WEIGHTS = {
        "imidazole": 1.0, "triazole": 1.0, "azole_general": 0.8,
        "tertiary_amine": 0.6, "furan": 0.7, "terminal_acetylene": 0.9,
        "benzimidazole": 0.8, "ritonavir_motif": 1.0, "grapefruit_furanocoumarins": 0.9,
        "calcium_channel_blocker": 0.6, "macrolide_lactone": 0.7, "hiv_protease_inhibitor": 0.8,
        "quinoline_antimalarial": 0.5  # 매우 특이적이지만 제한된 적용
    }

    SUBSTRATE_WEIGHTS = {
        "n_dealkylation": 1.0, "o_dealkylation_aromatic": 0.9, "o_dealkylation_aliphatic": 0.8,
        "benzylic_hydroxylation": 0.9, "allylic_hydroxylation": 0.8, "aliphatic_hydroxylation": 0.5,
        "epoxidation_aliphatic": 0.4, "s_oxidation": 0.8, "n_oxidation": 0.7,
        "aromatic_para_hydroxylation": 0.3, "tertiary_carbon_hydroxylation": 0.8,
        "steroid_6beta_hydroxylation": 0.9, "steroid_alpha_methyl": 0.8,
        "cyp3a4_preferred_size": 0.4, "lipophilic_aromatic": 0.3, "calcium_channel_substrate": 0.6
    }

    def __init__(self):
        self.rotate_smarts = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"

    @staticmethod
    def one_hot(x, allowable_set):
        return [x == s for s in (allowable_set if x in allowable_set else allowable_set[:-1] + [allowable_set[-1]])]


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

    def get_cyp3a4_features(self, mol):
        features = {}
        
        inhibitor_matches = 0
        for smarts in self.CYP3A4_INHIBITOR_SMARTS.values():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    inhibitor_matches += len(matches)
            except:
                continue
        
        substrate_matches = 0
        for smarts in self.CYP3A4_SUBSTRATE_SMARTS.values():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    substrate_matches += len(matches)
            except:
                continue
        
        features['inhibitor_pattern_count'] = min(inhibitor_matches / 10.0, 1.0)
        features['substrate_pattern_count'] = min(substrate_matches / 15.0, 1.0)
        features['inhibitor_ratio'] = inhibitor_matches / (inhibitor_matches + substrate_matches + 1)
        
        n_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
        n_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        n_sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
        n_halogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53])
        
        total_atoms = mol.GetNumAtoms()
        features['nitrogen_ratio'] = n_nitrogen / total_atoms
        features['oxygen_ratio'] = n_oxygen / total_atoms
        features['sulfur_ratio'] = n_sulfur / total_atoms
        features['halogen_ratio'] = n_halogens / total_atoms
        
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
    
    def get_feature(self, smiles):
        """Extract all molecular-level features including descriptors and fingerprints"""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Get all individual feature dictionaries
        physicochemical_features = self.get_physicochemical_features(mol)
        druglike_features = self.get_druglike_features(mol)
        cyp3a4_features = self.get_cyp3a4_features(mol)
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
        
        # Add CYP3A4 features
        cyp3a4_keys = [
            'inhibitor_pattern_count', 'substrate_pattern_count', 'inhibitor_ratio',
            'nitrogen_ratio', 'oxygen_ratio', 'sulfur_ratio', 'halogen_ratio'
        ]
        for key in cyp3a4_keys:
            all_descriptors.append(float(cyp3a4_features[key]))
        
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


    # Graph-related methods start here
    
    def get_ring_mappings(self, mol):
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
        degree_info = {}
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            neighbors = [n for n in atom.GetNeighbors()]
            
            total_degree = atom.GetDegree()                           # 총 연결도
            heavy_degree = len([n for n in neighbors if n.GetAtomicNum() > 1])  # 중원자 연결도
            
            # RDKit deprecation warning 해결
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()                    # 총 원자가
            
            neighbor_degrees = [n.GetDegree() for n in neighbors]     # 이웃 원자들의 연결도
            neighbor_heavy_degrees = [len([nn for nn in n.GetNeighbors() if nn.GetAtomicNum() > 1]) for n in neighbors]
            
            if neighbor_degrees:
                min_neighbor_deg = min(neighbor_degrees)              # 이웃 최소 연결도
                max_neighbor_deg = max(neighbor_degrees)              # 이웃 최대 연결도
                mean_neighbor_deg = sum(neighbor_degrees) / len(neighbor_degrees)  # 이웃 평균 연결도
                min_neighbor_heavy = min(neighbor_heavy_degrees)      # 이웃 최소 중원자 연결도
                max_neighbor_heavy = max(neighbor_heavy_degrees)      # 이웃 최대 중원자 연결도
                mean_neighbor_heavy = sum(neighbor_heavy_degrees) / len(neighbor_heavy_degrees)  # 이웃 평균 중원자 연결도
            else:
                min_neighbor_deg = max_neighbor_deg = mean_neighbor_deg = 0
                min_neighbor_heavy = max_neighbor_heavy = mean_neighbor_heavy = 0
            
            degree_centrality = total_degree / (mol.GetNumAtoms() - 1) if mol.GetNumAtoms() > 1 else 0  # 연결도 중심성
            
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

    def get_cyp3a4_features_for_graph(self, mol):
        """CYP3A4 특화 SMARTS 패턴 매칭 - 가중치 적용된 억제제와 기질 특징"""
        num_atoms = mol.GetNumAtoms()
        
        # 억제제 패턴 매칭 (가중치 적용)
        inhibitor_features = torch.zeros(num_atoms, len(self.CYP3A4_INHIBITOR_SMARTS))
        for idx, (name, smarts) in enumerate(self.CYP3A4_INHIBITOR_SMARTS.items()):
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    weight = self.INHIBITOR_WEIGHTS.get(name, 0.5)
                    for match in matches:
                        for atom_idx in match:
                            inhibitor_features[atom_idx, idx] = weight  # 가중치 적용
            except:
                continue
        
        # 기질 패턴 매칭 (가중치 적용)
        substrate_features = torch.zeros(num_atoms, len(self.CYP3A4_SUBSTRATE_SMARTS))
        for idx, (name, smarts) in enumerate(self.CYP3A4_SUBSTRATE_SMARTS.items()):
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    weight = self.SUBSTRATE_WEIGHTS.get(name, 0.5)
                    for match in matches:
                        for atom_idx in match:
                            substrate_features[atom_idx, idx] = weight  # 가중치 적용
            except:
                continue
        
        return inhibitor_features, substrate_features
    
    def get_stereochemistry_features(self, mol):
        """키랄성 및 입체화학 정보 추출"""
        num_atoms = mol.GetNumAtoms()
        stereo_features = torch.zeros(num_atoms, 8)
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            
            # 키랄 태그 정보
            chiral_tag = atom.GetChiralTag()
            if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                stereo_features[atom_idx, 0] = 1.0  # R 배치
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                stereo_features[atom_idx, 1] = 1.0  # S 배치
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                stereo_features[atom_idx, 2] = 1.0  # 미지정
                
            # 키랄 중심 가능성 (SP3 탄소 + 4개 다른 치환기)
            if (len(atom.GetNeighbors()) == 4 and 
                atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3):
                stereo_features[atom_idx, 3] = 1.0
                
            # E/Z 이성질체 참여
            for bond in atom.GetBonds():
                if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                    stereo_features[atom_idx, 4] = 1.0
                    break
                    
            # 평면성 관련
            if atom.GetIsAromatic():
                stereo_features[atom_idx, 5] = 1.0  # 방향족 평면성
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                stereo_features[atom_idx, 6] = 1.0  # SP2 평면성
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                stereo_features[atom_idx, 7] = 1.0  # SP 선형성
                
        return stereo_features
    
    def get_partial_charges(self, mol):
        """부분 전하 계산 (Gasteiger 방법) - 0-1 정규화"""
        num_atoms = mol.GetNumAtoms()
        
        # Gasteiger 전하 계산
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdPartialCharges.ComputeGasteigerCharges(mol)
        
        charges = torch.zeros(num_atoms, 2)  # 3 -> 2 차원으로 축소
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            charge = float(atom.GetProp('_GasteigerCharge'))
            
            # NaN 체크
            if torch.isnan(torch.tensor(charge)):
                charge = 0.0
                
            # 전하를 -1~+1 범위로 제한 후 0-1로 정규화
            charge = max(-1.0, min(1.0, charge))
            charges[atom_idx, 0] = (charge + 1.0) / 2.0  # -1~+1 -> 0~1 변환
            charges[atom_idx, 1] = abs(charge)  # 절댓값 (0-1)
            
        return charges
    
    def get_extended_neighborhood(self, mol):
        """확장된 근접 환경 정보 (2-hop, 3-hop 이웃) - 0-1 정규화"""
        num_atoms = mol.GetNumAtoms()
        ext_features = torch.zeros(num_atoms, 6)  # 12 -> 6 차원으로 축소
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            
            # 1-hop 이웃 정보
            neighbors_1 = list(atom.GetNeighbors())
            
            # 2-hop 이웃 정보
            neighbors_2 = set()
            for n1 in neighbors_1:
                for n2 in n1.GetNeighbors():
                    if n2.GetIdx() != atom_idx:
                        neighbors_2.add(n2)
            neighbors_2 = list(neighbors_2)
            
            # 1-hop 특성 (정규화)
            if neighbors_1:
                ext_features[atom_idx, 0] = min(len(neighbors_1) / 6.0, 1.0)  # 연결도 최대 6으로 제한
                ext_features[atom_idx, 1] = sum(n.GetIsAromatic() for n in neighbors_1) / len(neighbors_1)  # 방향족 비율
                ext_features[atom_idx, 2] = sum(1 for n in neighbors_1 if n.GetSymbol() in ['N', 'O', 'S']) / len(neighbors_1)  # 헤테로원자 비율
                
            # 2-hop 특성 (정규화)  
            if neighbors_2:
                ext_features[atom_idx, 3] = min(len(neighbors_2) / 20.0, 1.0)  # 2-hop 개수 최대 20으로 제한
                ext_features[atom_idx, 4] = sum(n.GetIsAromatic() for n in neighbors_2) / len(neighbors_2)  # 방향족 비율
                ext_features[atom_idx, 5] = sum(1 for n in neighbors_2 if n.GetSymbol() in ['N', 'O', 'S']) / len(neighbors_2)  # 헤테로원자 비율
                
        return ext_features
    
    def get_cyp3a4_binding_features(self, mol):
        """CYP3A4 활성 부위 특성을 고려한 분자 특징 (크기, 소수성, 유연성)"""
        num_atoms = mol.GetNumAtoms()
        binding_features = torch.zeros(num_atoms, 8)
        
        # 분자 크기와 형태 특성
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)  # Topological Polar Surface Area
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            
            # 1. CYP3A4 선호 크기 범위 (MW 300-800 Da)
            if 300 <= mol_weight <= 800:
                binding_features[atom_idx, 0] = 1.0 - abs(mol_weight - 550) / 250  # 550을 최적으로 가정
            
            # 2. 극성 표면적 (TPSA 60-140 Ų이 최적)
            if 60 <= tpsa <= 140:
                binding_features[atom_idx, 1] = 1.0 - abs(tpsa - 100) / 40
            
            # 3. 소수성 원자 (CYP3A4는 소수성 선호)
            if atom.GetSymbol() in ['C'] and not atom.GetIsAromatic():
                binding_features[atom_idx, 2] = 1.0
            elif atom.GetSymbol() in ['C'] and atom.GetIsAromatic():
                binding_features[atom_idx, 2] = 0.8
                
            # 4. 방향족 고리 (π-π 상호작용)
            if atom.GetIsAromatic():
                binding_features[atom_idx, 3] = 1.0
                
            # 5. 수소결합 공여체/수용체 (적당한 극성)
            if atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0:
                binding_features[atom_idx, 4] = 0.8  # 공여체
            elif atom.GetSymbol() in ['N', 'O'] and len(atom.GetNeighbors()) < 3:
                binding_features[atom_idx, 5] = 0.7  # 수용체
                
            # 6. 회전 가능한 결합 근처 (유연성)
            rotatable_neighbors = 0
            for bond in atom.GetBonds():
                if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and
                    not bond.IsInRing() and
                    bond.GetBeginAtom().GetAtomicNum() > 1 and
                    bond.GetEndAtom().GetAtomicNum() > 1):
                    rotatable_neighbors += 1
            binding_features[atom_idx, 6] = min(rotatable_neighbors / 4.0, 1.0)
            
            # 7. 전자 밀도가 높은 부위 (CYP 산화 위치)
            if (atom.GetSymbol() == 'C' and 
                len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]) >= 2):
                binding_features[atom_idx, 7] = 0.6
                
        return binding_features
    

    
    def get_ring_features(self, sizes, is_aromatic):
        is_in_ring = len(sizes) > 0                                  # 링 포함 여부
        num_rings = min(len(sizes), 4)                               # 링 개수 (최대 4)
        smallest = min(sizes) if sizes else 0                       # 최소 링 크기
        
        size_features = [False] * 6                                  # 링 크기별 특징 (3-8+ 원환)
        for size in sizes:
            if 3 <= size <= 8:
                size_features[size - 3] = True
            elif size > 8:
                size_features[5] = True                              # 8+ 원환
        
        return ([is_in_ring, is_aromatic, num_rings] + 
                size_features + 
                self.one_hot(num_rings, [0, 1, 2, 3, 4]) +
                self.one_hot(smallest, [0, 3, 4, 5, 6, 7, 8]))
    
    def get_atom_features(self, mol):
        atom_rings, _ = self.get_ring_mappings(mol)
        degree_info = self.get_degree_features(mol)
        inhibitor_feat, substrate_feat = self.get_cyp3a4_features_for_graph(mol)
        
        # 새로운 특성들 추가
        stereo_feat = self.get_stereochemistry_features(mol)
        charge_feat = self.get_partial_charges(mol)
        ext_neighbor_feat = self.get_extended_neighborhood(mol)
        binding_feat = self.get_cyp3a4_binding_features(mol)
        
        features = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            period, group = self.PERIODIC.get(symbol, (5, 18))
            electronegativity = self.ELECTRONEGATIVITY.get((period, group), 0.0)
            deg_info = degree_info[atom_idx]
            
            basic_feat = (
                self.one_hot(symbol, self.ATOMS) +                   # 원소 종류
                self.one_hot(period, self.PERIODS) +                 # 주기
                self.one_hot(group, self.GROUPS) +                   # 족
                [atom.GetIsAromatic(), atom.IsInRing(), 
                 min(atom.GetNumRadicalElectrons() / 3.0, 1.0),     # 라디칼 (최대 3으로 제한)
                 (atom.GetFormalCharge() + 3) / 6.0,                # 전하 (-3~+3 -> 0~1)
                 (electronegativity - 0.8) / 3.2]                   # 전기음성도 (0.8~4.0 -> 0~1)
            )
            
            # RDKit deprecation warning 해결
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()
                total_hs = atom.GetTotalNumHs()
            
            degree_feat = (
                self.one_hot(deg_info['total_degree'], self.DEGREES) +        # 총 연결도
                self.one_hot(deg_info['heavy_degree'], self.HEAVY_DEGREES) +  # 중원자 연결도
                self.one_hot(total_valence, self.VALENCES) +                  # 원자가
                self.one_hot(total_hs, self.TOTAL_HS) +                       # 수소 개수
                self.one_hot(atom.GetHybridization(), self.HYBRIDS) +         # 혼성화
                [deg_info['min_neighbor_deg'] / 6, deg_info['max_neighbor_deg'] / 6,    # 이웃 연결도 min/max
                 deg_info['mean_neighbor_deg'] / 6, deg_info['min_neighbor_heavy'] / 6,  # 이웃 연결도 평균, 중원자 min
                 deg_info['max_neighbor_heavy'] / 6, deg_info['mean_neighbor_heavy'] / 6, # 중원자 max/평균
                 deg_info['degree_centrality'], deg_info['degree_variance'] / 10]        # 중심성, 분산
            )
            
            ring_feat = self.get_ring_features(atom_rings[atom_idx], atom.GetIsAromatic())
            features.append(basic_feat + degree_feat + ring_feat)
        
        # 기본 SMARTS 패턴 매칭
        basic_smarts_feat = torch.zeros(mol.GetNumAtoms(), len(self.BASIC_SMARTS))
        for idx, smarts in enumerate(self.BASIC_SMARTS.values()):
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if matches:
                basic_smarts_feat[list(sum(matches, ())), idx] = 1
        
        atom_feat = torch.tensor(features, dtype=torch.float32)
        return torch.cat([atom_feat, basic_smarts_feat, inhibitor_feat, substrate_feat, 
                         stereo_feat, charge_feat, ext_neighbor_feat, binding_feat], dim=-1)
    
    def get_bond_features(self, mol):
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
                self.one_hot(bond.GetBondType(), self.BONDS) +       # 결합 종류
                self.one_hot(bond.GetStereo(), self.STEREOS) +       # 입체화학
                [bond.IsInRing(), bond.GetIsConjugated(), (src, dst) in rotatable_bonds]  # 링여부, 공액성, 회전가능성
            )
            
            bond_degree_feat = [
                abs(src_deg['total_degree'] - dst_deg['total_degree']) / 6,      # 연결도 차이
                abs(src_deg['heavy_degree'] - dst_deg['heavy_degree']) / 6,      # 중원자 연결도 차이
                abs(src_deg['valence'] - dst_deg['valence']) / 8,                # 원자가 차이
                (src_deg['total_degree'] + dst_deg['total_degree']) / 12,        # 연결도 합
                (src_deg['heavy_degree'] + dst_deg['heavy_degree']) / 12,        # 중원자 연결도 합
                (src_deg['valence'] + dst_deg['valence']) / 16,                  # 원자가 합
                abs(src_deg['degree_centrality'] - dst_deg['degree_centrality']), # 중심성 차이
                (src_deg['degree_centrality'] + dst_deg['degree_centrality']) / 2, # 중심성 평균
                min(src_deg['total_degree'], dst_deg['total_degree']) / 6,       # 연결도 최솟값
                max(src_deg['total_degree'], dst_deg['total_degree']) / 6        # 연결도 최댓값
            ]
            
            ring_feat = self.get_ring_features(bond_rings[bond.GetIdx()], bond.GetIsAromatic())
            
            adj[src, dst] = torch.tensor(basic_feat + bond_degree_feat + ring_feat, dtype=torch.float32)
        
        return adj + adj.transpose(0, 1)
    
    def get_graph(self, smiles):
        """Create molecular graph with node and edge features from SMILES"""
        mol = Chem.MolFromSmiles(smiles)

        atom_features = self.get_atom_features(mol)
        bond_features = self.get_bond_features(mol)

        src, dst = torch.where(bond_features.sum(dim=-1) > 0)
        g = dgl.graph((src, dst))

        g.ndata['feat'] = atom_features
        g.edata['feat'] = bond_features[src, dst]
        g.ndata['rwpe'] = dgl.random_walk_pe(g, 20)

        return g

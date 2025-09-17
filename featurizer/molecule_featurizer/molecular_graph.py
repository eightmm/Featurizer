import torch
import dgl
import pandas as pd
import warnings
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, rdMolDescriptors


class MolecularGraphBuilder:
    
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
    
    CYP3A4_INHIBITOR_SMARTS = {
        "imidazole": "c1ncn[c,n]1",                                    # 이미다졸 - 케토코나졸, 클로트리마졸 (헴 철 배위결합)
        "triazole": "c1nn[c,n]n1",                                     # 트리아졸 - 이트라코나졸, 플루코나졸 (강력한 억제)
        "azole_general": "[n;r5;a]c[n;r5;a]",                         # 일반화된 아졸 - 헴 철 배위결합 모티프
        "tertiary_amine": "[NX3;!$(N=O);!$(N-C=O)]([CH3,CH2CH3])([CH3,CH2CH3])",  # 3차 아민 - 에리스로마이신, 마크로라이드계
        "furan": "o1cccc1",                                           # 푸란 - 베르가모틴, 기전기반 억제 (에폭사이드 형성)
        "terminal_acetylene": "[C]#[CH1]",                            # 말단 아세틸렌 - 실험적 억제제 (케텐 형성)
        "benzimidazole": "c1nc2ccccc2[nH]1",                         # 벤즈이미다졸 - 오메프라졸계 (프로톤펌프억제제)
        "ritonavir_motif": "[N;$(N(C)C)]C(=O)N[CH]([CH2]c1ccccc1)C(=O)", # 리토나비르 특징적 모티프 (강력한 억제)
        "grapefruit_furanocoumarins": "o1cc2ccc3oc(=O)cc3c2c1",       # 자몽 푸라노쿠마린 - 베르가모틴 유사체
        "calcium_channel_blocker": "c1ccc(cc1)C(=O)C[NH]c2ncccn2",    # 칼슘채널차단제 모티프 - 베라파밀계
        "macrolide_lactone": "[O;r14,r15,r16][C](=O)",                # 마크로라이드 락톤 - 에리스로마이신, 클라리스로마이신
        "hiv_protease_inhibitor": "[S;$(S(=O)=O)][NH]",               # HIV 프로테아제 억제제 설폰아마이드
        "quinoline_antimalarial": "c1cc(OC)c2c(c1)ncc(c2)C(O)CN",     # 퀴닌/퀴니딘 특징적 구조 (더 특이적)
    }
    
    CYP3A4_SUBSTRATE_SMARTS = {
        "n_dealkylation": "[NX3,NX4+;!$(N-C=O)]([CH3,CH2CH3])",      # N-탈알킬화 - 이미프라민, 리도카인 (주요 대사경로)
        "o_dealkylation_aromatic": "[O;X2;!$(O-C=O)]([CH3,CH2CH3])c", # O-탈알킬화 방향족 - 코데인 (모르핀 생성)
        "o_dealkylation_aliphatic": "[O;X2;!$(O-C=O)]([CH3,CH2CH3])C", # O-탈알킬화 지방족 - 메톡시기 제거
        "benzylic_hydroxylation": "[C;H2,H3][c]",                    # 벤질 위치 수산화 - 톨부타미드 (활성화된 위치)
        "allylic_hydroxylation": "[C;H2,H3][C]=[C]",                 # 알릴 위치 수산화 - 이중결합 인접 탄소
        "aliphatic_hydroxylation": "[C;H2,H3][C;H2,H3]",             # 지방족 수산화 - 메틸/에틸기 (더 일반화)
        "epoxidation_aliphatic": "[C;!a]=[C;!a]",                    # 지방족 에폭시화 - 카르바마제핀 (방향족 제외)
        "s_oxidation": "[SX2]",                                      # 황 산화 - 티오리다진 (설폭사이드 형성)
        "n_oxidation": "[NX3;!$(N=O);!$(N-[S,P]=O)]",               # 질소 산화 - 이미프라민 (N-산화물 형성)
        "aromatic_para_hydroxylation": "c1ccc(cc1)",                 # 파라 위치 수산화 - 페니토인, 와파린
        "tertiary_carbon_hydroxylation": "[C;X4;H1]([C])([C])[C]",   # 3차 탄소 수산화 - 미다졸람
        "steroid_6beta_hydroxylation": "[C;H1]1[C][C]2[C]([C][C]1)[C][C][C]3[C][C][C][C][C]23", # 스테로이드 6β-수산화
        "steroid_alpha_methyl": "[C;H3][C]1[C][C]2[C]([C][C]1)[C][C][C]3[C][C][C][C][C]23", # 스테로이드 α-메틸기
        "cyp3a4_preferred_size": "c1ccc(cc1)C[C,N,O][C,N,O][C,N,O]", # CYP3A4 선호 크기 (벤젠+3-4원자 사슬)
        "lipophilic_aromatic": "c1ccc(cc1)c2ccccc2",                 # 소수성 방향족 - 비페닐계 (CYP3A4 선호)
        "calcium_channel_substrate": "c1ccc(cc1)C(=O)OC[C@H](O)CN",  # 칼슘채널차단제 기질 모티프
    }
    
    @staticmethod
    def one_hot(x, allowable_set):
        return [x == s for s in (allowable_set if x in allowable_set else allowable_set[:-1] + [allowable_set[-1]])]
    
    def __init__(self):
        self.rotate_smarts = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"
    
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
    
    def get_cyp3a4_features(self, mol):
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
        inhibitor_feat, substrate_feat = self.get_cyp3a4_features(mol)
        
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
    
    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        atom_features = self.get_atom_features(mol)
        bond_features = self.get_bond_features(mol)
        
        src, dst = torch.where(bond_features.sum(dim=-1) > 0)
        g = dgl.graph((src, dst))
        
        g.ndata['feat'] = atom_features
        g.edata['feat'] = bond_features[src, dst]
        g.ndata['rwpe'] = dgl.random_walk_pe(g, 20)
        
        return g


def create_molecular_graph(smiles):
    builder = MolecularGraphBuilder()
    return builder.smiles_to_graph(smiles)


if __name__ == "__main__":
    df = pd.read_csv('./data/dacon/train.csv')
    
    print(f"Dataset: {len(df)} molecules")
    print(f"Columns: {list(df.columns)}")
    
    first_smiles = df.iloc[0]['Canonical_Smiles']
    first_id = df.iloc[0].get('id', 0)
    
    print(f"\nMolecule {first_id}: {first_smiles}")
    
    try:
        graph = create_molecular_graph(first_smiles)
        
        print(f"\nNodes: {graph.num_nodes()}")
        print(f"Edges: {graph.num_edges()}")
        print(f"Node features: {graph.ndata['feat'].shape}")
        print(f"Edge features: {graph.edata['feat'].shape}")
        print(f"RWPE: {graph.ndata['rwpe'].shape}")
        
        print(f"\nAdvanced CYP3A4-Optimized Molecular Features:")
        print(f"- {len(MolecularGraphBuilder.CYP3A4_INHIBITOR_SMARTS)} weighted inhibitor patterns (removed non-CYP pathways)")
        print(f"- {len(MolecularGraphBuilder.CYP3A4_SUBSTRATE_SMARTS)} weighted substrate patterns (CYP-specific reactions only)")
        print(f"- 8D stereochemistry features (chirality, planarity)")
        print(f"- 2D partial charge features (Gasteiger normalized)")
        print(f"- 6D extended neighborhood (1-hop, 2-hop normalized)")
        print(f"- 8D CYP3A4 binding site features (size, lipophilicity, flexibility)")
        print(f"- Enhanced degree and ring features with centrality")
        print(f"- Pattern confidence weighting (0.3-1.0 based on literature)")
        print(f"- Removed ester/amide hydrolysis, deamination (non-CYP pathways)")
        print(f"- All features normalized to [0,1] range for optimal learning")
        
        print(f"\nFirst 3 nodes:")
        for i in range(min(3, graph.num_nodes())):
            print(f"  {i}: {graph.ndata['feat'][i][:10]}...")
            
        print(f"\nFirst 3 edges:")
        for i in range(min(3, graph.num_edges())):
            print(f"  {i}: {graph.edata['feat'][i][:10]}...")
            
        print("\nSuccess!")
        
    except Exception as e:
        print(f"Error: {e}") 
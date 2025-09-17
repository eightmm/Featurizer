#!/usr/bin/env python3
"""
Advanced usage examples of the Featurizer package.
"""

import os
import sys
from pathlib import Path
import torch
from rdkit import Chem

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from featurizer import (
    MoleculeFeatureExtractor,
    MoleculeGraphBuilder,
    create_molecule_graph,
    ProteinFeaturizer,
    PDBStandardizer,
    ResidueFeaturizer
)


def advanced_molecule_features():
    """
    Demonstrate advanced molecule feature extraction with detailed analysis.
    """
    print("\n" + "=" * 60)
    print(" Advanced Molecule Feature Extraction")
    print("=" * 60)

    # Example molecules with different properties
    molecules = {
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Penicillin": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
        "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
    }

    extractor = MoleculeFeatureExtractor()

    for name, smiles in molecules.items():
        print(f"\n{'='*40}")
        print(f"Analyzing: {name}")
        print(f"SMILES: {smiles}")
        print(f"{'='*40}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Failed to parse SMILES for {name}")
            continue

        # Get individual feature categories
        phys_features = extractor.get_physicochemical_features(mol)
        drug_features = extractor.get_druglike_features(mol)
        struct_features = extractor.get_structural_features(mol)
        atom_comp_features = extractor.get_atom_composition_features(mol)

        # Display key features
        print("\n📊 Physicochemical Properties:")
        print(f"  - Molecular Weight: {phys_features['mw']*1000:.2f} Da")
        print(f"  - LogP: {(phys_features['logp']*10)-5:.2f}")
        print(f"  - TPSA: {phys_features['tpsa']*200:.2f} Ų")
        print(f"  - Rotatable Bonds: {int(phys_features['n_rotatable_bonds']*20)}")

        print("\n💊 Drug-likeness:")
        print(f"  - Lipinski Violations: {int(drug_features['lipinski_violations']*4)}")
        print(f"  - QED Score: {drug_features['qed']:.3f}")
        print(f"  - Fraction Csp3: {drug_features['frac_csp3']:.3f}")

        print("\n🔬 Structural Features:")
        print(f"  - Number of Rings: {int(struct_features['n_ring_systems']*8)}")
        print(f"  - Max Ring Size: {int(struct_features['max_ring_size']*12)}")

        print("\n⚛️ Atom Composition:")
        print(f"  - N ratio: {atom_comp_features['nitrogen_ratio']:.3f}")
        print(f"  - O ratio: {atom_comp_features['oxygen_ratio']:.3f}")
        print(f"  - Halogen ratio: {atom_comp_features['halogen_ratio']:.3f}")

        # Get all features including fingerprints
        all_features = extractor.extract_all_features(mol, add_hs=True)

        print("\n🔑 Fingerprint Sizes:")
        for fp_name, fp_tensor in all_features.items():
            if fp_name != 'descriptor':
                print(f"  - {fp_name}: {fp_tensor.shape}")


def molecule_graph_features():
    """
    Demonstrate molecule graph feature extraction.
    """
    print("\n" + "=" * 60)
    print(" Molecule Graph Features")
    print("=" * 60)

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    print(f"\nCreating graph for: {smiles}")

    try:
        # Create molecule graph
        mol_graph = create_molecule_graph(smiles)

        print("\n📈 Graph Structure:")
        print(f"  - Number of nodes: {mol_graph.num_nodes()}")
        print(f"  - Number of edges: {mol_graph.num_edges()}")

        # Alternative method using class directly
        graph_builder = MoleculeGraphBuilder()
        mol = Chem.MolFromSmiles(smiles)
        graph_data = graph_builder.smiles_to_graph(smiles)

        print("\n🔍 Detailed Graph Info:")
        print(f"  - Node features shape: {graph_data.ndata['feat'].shape if 'feat' in graph_data.ndata else 'N/A'}")
        print(f"  - Edge features shape: {graph_data.edata['feat'].shape if 'feat' in graph_data.edata else 'N/A'}")
    except ImportError:
        print("⚠ DGL not installed, skipping graph features")
    except Exception as e:
        print(f"⚠ Graph creation skipped: {e}")


def advanced_protein_features():
    """
    Demonstrate advanced protein feature extraction.
    """
    print("\n" + "=" * 60)
    print(" Advanced Protein Feature Extraction")
    print("=" * 60)

    # Example: Manual pipeline control
    pdb_file = "protein.pdb"
    clean_pdb = "protein_clean.pdb"

    if not os.path.exists(pdb_file):
        print(f"⚠ Example PDB file '{pdb_file}' not found")
        print("  Creating a minimal example PDB...")

        # Create a minimal PDB for demonstration
        minimal_pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.221   2.370   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       2.000  -0.800   1.200  1.00 20.00           C
ATOM      6  N   GLY A   2       3.310   1.590   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       3.960   2.890   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.480   2.760   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.040   1.670   0.000  1.00 20.00           O
END
"""
        with open(pdb_file, 'w') as f:
            f.write(minimal_pdb)
        print("  ✓ Created minimal example PDB")

    try:
        # Step 1: Standardize PDB
        print("\n1️⃣ Standardizing PDB file...")
        standardizer = PDBStandardizer(remove_hydrogens=True)
        standardizer.standardize(pdb_file, clean_pdb)
        print("  ✓ PDB standardized")

        # Step 2: Extract features manually
        print("\n2️⃣ Extracting residue features...")
        featurizer = ResidueFeaturizer(clean_pdb)
        node_features, edge_features = featurizer.get_features()

        print("  ✓ Features extracted")
        print(f"    - Nodes: {node_features['coord'].shape[0]} residues")
        print(f"    - Node scalar features: {len(node_features['node_scalar_features'])} types")
        print(f"    - Node vector features: {len(node_features['node_vector_features'])} types")
        print(f"    - Edges: {len(edge_features['edges'][0])} connections")

        # Clean up
        if os.path.exists(clean_pdb):
            os.remove(clean_pdb)

    except Exception as e:
        print(f"❌ Error during protein processing: {str(e)}")


def feature_comparison():
    """
    Compare features between molecules and proteins.
    """
    print("\n" + "=" * 60)
    print(" Molecule vs Protein Feature Comparison")
    print("=" * 60)

    # Compare molecule features
    print("\n🧪 Molecule Features:")
    drug_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C"  # Ibuprofen

    extractor = MoleculeFeatureExtractor()
    mol = Chem.MolFromSmiles(drug_molecule)

    if mol:
        drug_features = extractor.get_druglike_features(mol)
        phys_features = extractor.get_physicochemical_features(mol)

        print(f"  Ibuprofen (SMILES: {drug_molecule[:20]}...):")
        print(f"    - MW: {phys_features['mw']*1000:.1f} Da")
        print(f"    - LogP: {(phys_features['logp']*10)-5:.2f}")
        print(f"    - Lipinski violations: {int(drug_features['lipinski_violations']*4)}")
        print(f"    - QED score: {drug_features['qed']:.3f}")

    # Compare protein features
    print("\n🧬 Protein Features:")
    print("  Typical protein characteristics:")
    print("    - Residue-based representation")
    print("    - 3D structural features")
    print("    - Interaction networks")
    print("    - SASA calculations")
    print("    - Secondary structure elements")


def memory_efficient_batch_processing():
    """
    Demonstrate memory-efficient processing for large datasets.
    """
    print("\n" + "=" * 60)
    print(" Memory-Efficient Batch Processing")
    print("=" * 60)

    # Example: Processing molecules in batches
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "CC(C)C",  # Isobutane
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    ]

    print("\n📦 Batch Processing Molecules:")

    batch_size = 2
    all_features = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        print(f"\n  Processing batch {i//batch_size + 1}:")

        batch_features = []
        for smiles in batch:
            try:
                from featurizer import create_molecule_features
                features = create_molecule_features(smiles, add_hs=False)
                batch_features.append(features['descriptor'])
                print(f"    ✓ {smiles}")
            except Exception as e:
                print(f"    ❌ {smiles}: {str(e)}")

        if batch_features:
            # Stack features for batch
            batch_tensor = torch.stack(batch_features)
            all_features.append(batch_tensor)

    # Combine all batches
    if all_features:
        final_features = torch.cat(all_features, dim=0)
        print(f"\n✓ Total features shape: {final_features.shape}")


def custom_feature_pipeline():
    """
    Create a custom feature extraction pipeline combining molecule and protein features.
    """
    print("\n" + "=" * 60)
    print(" Custom Feature Pipeline")
    print("=" * 60)

    class UnifiedFeaturePipeline:
        def __init__(self, normalize=True):
            self.normalize = normalize
            self.molecule_extractor = MoleculeFeatureExtractor()
            self.protein_featurizer = ProteinFeaturizer()

        def process_molecule(self, smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Get molecule features
            features = self.molecule_extractor.extract_all_features(mol)

            if self.normalize:
                # Normalize descriptors to [0, 1]
                desc = features['descriptor']
                desc_min = desc.min()
                desc_max = desc.max()
                if desc_max > desc_min:
                    features['descriptor'] = (desc - desc_min) / (desc_max - desc_min)

            features['type'] = 'molecule'
            return features

        def process_protein(self, pdb_file):
            # Get protein features
            features = self.protein_featurizer.extract(pdb_file)
            features['type'] = 'protein'
            return features

    # Use custom pipeline
    pipeline = UnifiedFeaturePipeline(normalize=True)

    # Process a molecule
    smiles = "CC(C)CC1=CC=C(C=C1)C(C)C"  # Ibuprofen
    mol_features = pipeline.process_molecule(smiles)

    print(f"\n✓ Custom pipeline for molecules:")
    print(f"  - Type: {mol_features['type']}")
    print(f"  - Normalized descriptors: {mol_features['descriptor'].shape}")
    print(f"  - Min value: {mol_features['descriptor'].min():.3f}")
    print(f"  - Max value: {mol_features['descriptor'].max():.3f}")

    print(f"\n✓ Custom pipeline for proteins:")
    print(f"  - Accepts PDB files")
    print(f"  - Returns node and edge features")
    print(f"  - Includes metadata")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" Featurizer Package - Advanced Usage Examples")
    print("=" * 70)

    # Run all advanced examples
    advanced_molecule_features()
    molecule_graph_features()
    advanced_protein_features()
    feature_comparison()
    memory_efficient_batch_processing()
    custom_feature_pipeline()

    print("\n" + "=" * 70)
    print(" Advanced examples completed!")
    print("=" * 70)
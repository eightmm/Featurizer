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
    MoleculeFeaturizer,
    ProteinFeaturizer
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

    for name, smiles in molecules.items():
        print(f"\n{'='*40}")
        print(f"Analyzing: {name}")
        print(f"SMILES: {smiles}")
        print(f"{'='*40}")

        try:
            # Initialize featurizer with molecule
            featurizer = MoleculeFeaturizer(smiles)

            # Extract comprehensive features
            features = featurizer.get_feature()
        except Exception as e:
            print(f"⚠ Could not process {name}: {e}")
            continue

        # Display molecular descriptors
        descriptors = features['descriptor']
        print(f"\n📊 Molecular Descriptors (shape: {descriptors.shape}):")
        print(f"  - MW: {descriptors[0]*1000:.1f} Da")
        print(f"  - LogP: {(descriptors[1]*10)-5:.2f}")
        print(f"  - TPSA: {descriptors[2]*200:.1f} Ų")
        print(f"  - Rotatable bonds: {int(descriptors[3]*20)}")
        print(f"  - H-bond donors: {int(descriptors[5]*10)}")
        print(f"  - H-bond acceptors: {int(descriptors[6]*15)}")

        # Display fingerprints
        print(f"\n🔬 Molecular Fingerprints:")
        for fp_name, fp_tensor in features.items():
            if fp_name != 'descriptor':
                print(f"  - {fp_name}: shape {fp_tensor.shape}, "
                      f"density: {(fp_tensor > 0).sum().item() / fp_tensor.numel():.3f}")

        # Drug-likeness assessment
        lipinski = descriptors[35]  # Lipinski violations
        qed = descriptors[37]  # QED score

        print(f"\n💊 Drug-likeness:")
        print(f"  - Lipinski violations: {int(lipinski*4)}")
        print(f"  - QED score: {qed:.3f}")
        print(f"  - Assessment: {'Good' if lipinski < 0.25 and qed > 0.5 else 'Poor'}")

    # Demonstrate graph features
    smiles = "c1ccc(cc1)CC(C(=O)O)N"  # Phenylalanine
    print(f"\n\n{'='*60}")
    print(" Graph Representation Example")
    print(f"{'='*60}")
    print(f"Molecule: Phenylalanine")
    print(f"SMILES: {smiles}")

    # Create molecule graph using new API
    try:
        featurizer = MoleculeFeaturizer(smiles)
        node, edge = featurizer.get_graph()

        print("\n📈 Graph Structure:")
        print(f"  - Number of nodes: {node['node_feats'].shape[0]}")
        print(f"  - Number of edges: {edge['edges'].shape[1]}")
        print(f"  - Node features shape: {node['node_feats'].shape}")
        print(f"  - Edge features shape: {edge['edge_feats'].shape}")

        if 'coords' in node:
            print(f"  - 3D coordinates available: {node['coords'].shape}")
    except Exception as e:
        print(f"⚠ Graph creation skipped: {e}")


def advanced_protein_features():
    """
    Demonstrate advanced protein feature extraction.
    """
    print("\n" + "=" * 60)
    print(" Advanced Protein Feature Extraction")
    print("=" * 60)

    # Create example PDB file (small peptide)
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.221   2.370   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.988  -0.814  -1.196  1.00 20.00           C
ATOM      6  N   VAL A   2       3.337   1.590   0.000  1.00 20.00           N
ATOM      7  CA  VAL A   2       3.980   2.897   0.000  1.00 20.00           C
ATOM      8  C   VAL A   2       5.499   2.786   0.000  1.00 20.00           C
ATOM      9  O   VAL A   2       6.150   1.736   0.000  1.00 20.00           O
ATOM     10  CB  VAL A   2       3.518   3.733   1.196  1.00 20.00           C
END"""

    # Save temporary PDB file
    temp_pdb = "temp_peptide.pdb"
    with open(temp_pdb, 'w') as f:
        f.write(pdb_content)

    try:
        featurizer = ProteinFeaturizer(temp_pdb)

        # Get residue-level features
        print("\n🔬 Residue-Level Features:")
        res_node, res_edge = featurizer.get_residue_features()

        print(f"  - Number of residues: {res_node['coord'].shape[0]}")
        print(f"  - Scalar features shape: {res_node['node_scalar_features'].shape}")
        print(f"  - Vector features shape: {res_node['node_vector_features'].shape}")
        print(f"  - Edges (8Å cutoff): {res_edge['edges'][0].shape[0] if res_edge['edges'][0].shape[0] > 0 else 0}")

        # Get atom-level features
        print("\n⚛️ Atom-Level Features:")
        atom_node, atom_edge = featurizer.get_atom_features()

        print(f"  - Number of atoms: {atom_node['coord'].shape[0]}")
        print(f"  - Atom tokens: {atom_node['atom_tokens'].shape}")
        print(f"  - SASA per atom: {atom_node['sasa'].sum():.2f} Ų total")
        print(f"  - Edges (4Å cutoff): {atom_edge['edges'][0].shape[0]}")

        # Get specific feature types
        print("\n📐 Geometric Features:")
        geometry = featurizer.get_residue_geometry()
        for res_idx in range(min(2, len(geometry['phi']))):
            print(f"  Residue {res_idx+1}:")
            print(f"    - Phi: {geometry['phi'][res_idx]:.2f}°")
            print(f"    - Psi: {geometry['psi'][res_idx]:.2f}°")

    except Exception as e:
        print(f"⚠ Protein feature extraction failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)


def custom_smarts_features():
    """
    Demonstrate custom SMARTS pattern matching.
    """
    print("\n" + "=" * 60)
    print(" Custom SMARTS Pattern Features")
    print("=" * 60)

    # Define pharmacophore patterns
    pharmacophore_patterns = {
        'h_donor': '[NX3,NX4+][H]',
        'h_acceptor': '[O,N;!H0]',
        'aromatic': 'a',
        'halogen': '[F,Cl,Br,I]',
        'positive': '[*+]',
        'negative': '[*-]'
    }

    # Test molecule
    smiles = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    print(f"\nMolecule: Paracetamol")
    print(f"SMILES: {smiles}")

    # Initialize with custom patterns
    featurizer = MoleculeFeaturizer(smiles, custom_smarts=pharmacophore_patterns)

    # Get graph with custom features
    node, edge = featurizer.get_graph()

    print(f"\n📊 Node Features with Custom SMARTS:")
    print(f"  - Base features: 122 dimensions")
    print(f"  - Custom patterns: {len(pharmacophore_patterns)} dimensions")
    print(f"  - Total dimensions: {node['node_feats'].shape[1]}")

    # Get custom features separately for analysis
    custom_feats = featurizer.get_custom_smarts_features()
    if custom_feats:
        print(f"\n🎯 Pattern Matches:")
        for i, name in enumerate(custom_feats['names']):
            matching_atoms = torch.where(custom_feats['features'][:, i] > 0)[0]
            if len(matching_atoms) > 0:
                print(f"  - {name}: atoms {matching_atoms.tolist()}")


def comparison_features():
    """
    Compare molecule vs protein feature extraction.
    """
    print("\n" + "=" * 60)
    print(" Molecule vs Protein Features Comparison")
    print("=" * 60)

    # Drug molecule example
    drug_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen

    print("\n💊 Molecule Features:")
    try:
        mol_featurizer = MoleculeFeaturizer(drug_molecule)
        mol_features = mol_featurizer.get_feature()

        # Physical-chemical features
        phys_features = mol_features['descriptor']
        drug_features = mol_features['descriptor']

        print(f"  Ibuprofen (SMILES: {drug_molecule[:20]}...):")
        print(f"    - MW: {phys_features[0]*1000:.1f} Da")
        print(f"    - LogP: {(phys_features[1]*10)-5:.2f}")
        print(f"    - Lipinski violations: {int(drug_features[35]*4)}")
        print(f"    - QED score: {drug_features[37]:.3f}")
    except Exception as e:
        print(f"  ⚠ Could not process molecule: {e}")

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
                featurizer = MoleculeFeaturizer(smiles)
                features = featurizer.get_feature()
                batch_features.append(features['descriptor'])
                print(f"    ✓ {smiles}")
            except Exception as e:
                print(f"    ✗ {smiles}: {e}")

        if batch_features:
            # Stack features for batch processing
            batch_tensor = torch.stack(batch_features)
            all_features.append(batch_tensor)

    # Combine all batches
    if all_features:
        combined = torch.cat(all_features, dim=0)
        print(f"\n📊 Combined Features Shape: {combined.shape}")
        print(f"   Mean MW: {combined[:, 0].mean()*1000:.1f} Da")
        print(f"   Mean LogP: {(combined[:, 1].mean()*10)-5:.2f}")


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 80)
    print(" " * 20 + "ADVANCED FEATURIZER EXAMPLES")
    print("=" * 80)

    # Run examples
    advanced_molecule_features()
    advanced_protein_features()
    custom_smarts_features()
    comparison_features()
    memory_efficient_batch_processing()

    print("\n" + "=" * 80)
    print(" " * 30 + "EXAMPLES COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
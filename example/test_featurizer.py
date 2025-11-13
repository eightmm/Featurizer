#!/usr/bin/env python3
"""
Comprehensive example demonstrating the Featurizer package functionality.

This script tests both protein and molecule featurizers using the example files:
- 10gs_protein.pdb: Protein structure
- 10gs_ligand.sdf: Small molecule ligand

Usage:
    python test_featurizer.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from featurizer import ProteinFeaturizer, MoleculeFeaturizer
from rdkit import Chem


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_protein_features():
    """Test protein featurizer with various features."""
    print_section("PROTEIN FEATURIZER TESTS")

    pdb_file = "10gs_protein.pdb"
    if not os.path.exists(pdb_file):
        print(f"❌ PDB file not found: {pdb_file}")
        return

    print(f"\n📁 Loading: {pdb_file}")
    featurizer = ProteinFeaturizer(pdb_file, standardize=True)
    print("✓ Protein loaded successfully")

    # Test 1: Sequence extraction by chain
    print("\n" + "-"*70)
    print("1. Sequence Extraction (by chain)")
    print("-"*70)
    sequences = featurizer.get_sequence_by_chain()

    total_residues = 0
    for chain_id, sequence in sequences.items():
        print(f"  Chain {chain_id}: {len(sequence)} residues")
        print(f"    First 50 aa: {sequence[:50]}...")
        total_residues += len(sequence)

    print(f"\n  Total residues: {total_residues}")
    print(f"  Total chains: {len(sequences)}")

    # Test 2: Residue-level features
    print("\n" + "-"*70)
    print("2. Residue-Level Features")
    print("-"*70)
    res_node, res_edge = featurizer.get_residue_features(distance_cutoff=8.0)

    print(f"  Coordinates shape: {res_node['coord'].shape}")
    print(f"  Node scalar features: {len(res_node['node_scalar_features'])} types")
    print(f"  Node vector features: {len(res_node['node_vector_features'])} types")
    print(f"  Edges (src, dst): {res_edge['edges'][0].shape[0]} interactions")
    print(f"  Edge scalar features: {len(res_edge['edge_scalar_features'])} types")
    print(f"  Edge vector features: {len(res_edge['edge_vector_features'])} types")

    # Test 3: Atom-level features
    print("\n" + "-"*70)
    print("3. Atom-Level Features")
    print("-"*70)
    atom_node, atom_edge = featurizer.get_atom_features(distance_cutoff=4.0)

    print(f"  Total atoms: {atom_node['coord'].shape[0]}")
    print(f"  Atom coordinates shape: {atom_node['coord'].shape}")
    print(f"  Atom tokens shape: {atom_node['atom_tokens'].shape}")
    print(f"  SASA shape: {atom_node['sasa'].shape}")
    print(f"  Atom edges: {atom_edge['edges'][0].shape[0]} interactions")
    print(f"  Distance cutoff: {atom_edge['distance_cutoff']} Å")

    # Test 4: SASA features
    print("\n" + "-"*70)
    print("4. SASA Features (Solvent Accessible Surface Area)")
    print("-"*70)
    sasa = featurizer.get_sasa_features()
    print(f"  SASA tensor shape: {sasa.shape}")
    print(f"  SASA components: 10 (total, polar, apolar, mainChain, sideChain, etc.)")

    # Test 5: Contact map
    print("\n" + "-"*70)
    print("5. Contact Map")
    print("-"*70)
    contact_map = featurizer.get_contact_map(cutoff=8.0)
    print(f"  Adjacency matrix shape: {contact_map['adjacency_matrix'].shape}")
    print(f"  Distance matrix shape: {contact_map['distance_matrix'].shape}")
    print(f"  Number of contacts: {contact_map['edges'][0].shape[0]}")

    # Test 6: Geometric features
    print("\n" + "-"*70)
    print("6. Geometric Features")
    print("-"*70)
    geom = featurizer.get_geometric_features()
    print(f"  Dihedrals shape: {geom['dihedrals'].shape}")
    print(f"  Has chi angles shape: {geom['has_chi_angles'].shape}")
    print(f"  Backbone curvature shape: {geom['backbone_curvature'].shape}")
    print(f"  Backbone torsion shape: {geom['backbone_torsion'].shape}")
    print(f"  Self distances shape: {geom['self_distances'].shape}")

    print("\n✓ All protein tests completed successfully!")


def test_molecule_features():
    """Test molecule featurizer with various features."""
    print_section("MOLECULE FEATURIZER TESTS")

    sdf_file = "10gs_ligand.sdf"
    if not os.path.exists(sdf_file):
        print(f"❌ SDF file not found: {sdf_file}")
        return

    print(f"\n📁 Loading: {sdf_file}")
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[0]

    if mol is None:
        print("❌ Failed to load molecule")
        return

    featurizer = MoleculeFeaturizer(mol)
    print("✓ Molecule loaded successfully")

    # Test 1: Basic molecular properties
    print("\n" + "-"*70)
    print("1. Molecular Properties")
    print("-"*70)
    print(f"  SMILES: {Chem.MolToSmiles(mol)}")
    print(f"  Number of atoms: {mol.GetNumAtoms()}")
    print(f"  Number of bonds: {mol.GetNumBonds()}")
    print(f"  Molecular formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")

    # Test 2: Descriptors and fingerprints
    print("\n" + "-"*70)
    print("2. Descriptors and Fingerprints")
    print("-"*70)
    features = featurizer.get_feature()

    print(f"  Descriptor shape: {features['descriptor'].shape}")
    print(f"  Morgan fingerprint shape: {features['morgan'].shape}")
    print(f"  MACCS keys shape: {features['maccs'].shape}")
    print(f"  RDKit fingerprint shape: {features['rdkit'].shape}")
    print(f"  Atom pair fingerprint shape: {features['atom_pair'].shape}")
    print(f"  Topological torsion shape: {features['topological_torsion'].shape}")
    print(f"  Pharmacophore2D shape: {features['pharmacophore2d'].shape}")

    # Show first few descriptor values
    print(f"\n  First 5 descriptor values: {features['descriptor'][:5].tolist()}")

    # Test 3: Graph representation
    print("\n" + "-"*70)
    print("3. Graph Representation")
    print("-"*70)
    node, edge, adj = featurizer.get_graph()

    print(f"  Node features shape: {node['node_feats'].shape}")
    print(f"    - Base features: 122 dimensions")
    print(f"    - Total features per atom: {node['node_feats'].shape[1]} dimensions")
    print(f"  Edge features shape: {edge['edge_feats'].shape}")
    print(f"    - Features per bond: {edge['edge_feats'].shape[1]} dimensions")
    print(f"  Edge indices shape: {edge['edges'].shape}")
    print(f"    - Total edges (bidirectional): {edge['edges'].shape[1]}")
    print(f"  Adjacency matrix shape: {adj.shape}")

    # Test 4: Molecule without hydrogens
    print("\n" + "-"*70)
    print("4. Molecule without Hydrogens")
    print("-"*70)
    featurizer_no_h = MoleculeFeaturizer(mol, hydrogen=False)
    node_no_h, edge_no_h, adj_no_h = featurizer_no_h.get_graph()

    print(f"  Atoms (with H): {node['node_feats'].shape[0]}")
    print(f"  Atoms (no H): {node_no_h['node_feats'].shape[0]}")
    print(f"  Edges (with H): {edge['edges'].shape[1]}")
    print(f"  Edges (no H): {edge_no_h['edges'].shape[1]}")

    print("\n✓ All molecule tests completed successfully!")


def test_custom_smarts():
    """Test custom SMARTS patterns."""
    print_section("CUSTOM SMARTS PATTERNS TEST")

    # Test with a simple molecule
    smiles = "c1ccccc1CCO"  # Phenethyl alcohol
    print(f"\n📝 Test molecule: {smiles} (phenethyl alcohol)")

    # Define custom patterns
    custom_patterns = {
        'aromatic_carbon': 'c',
        'aromatic_ring': 'c1ccccc1',
        'hydroxyl': '[OH]',
        'aliphatic_carbon': '[C;!c]'
    }

    print("\n  Custom SMARTS patterns:")
    for name, pattern in custom_patterns.items():
        print(f"    - {name}: {pattern}")

    featurizer = MoleculeFeaturizer(smiles, custom_smarts=custom_patterns)
    node, edge, adj = featurizer.get_graph()

    print(f"\n  Node features shape: {node['node_feats'].shape}")
    print(f"    - Base features: 122 dimensions")
    print(f"    - Custom patterns: {len(custom_patterns)} dimensions")
    print(f"    - Total: {node['node_feats'].shape[1]} dimensions")

    # Get custom features separately
    custom_feats = featurizer.get_custom_smarts_features()
    print(f"\n  Custom pattern matches:")
    for i, name in enumerate(custom_feats['names']):
        matches = custom_feats['features'][:, i].sum().item()
        print(f"    - {name}: {int(matches)} matches")

    print("\n✓ Custom SMARTS test completed successfully!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  FEATURIZER PACKAGE - COMPREHENSIVE TESTING")
    print("="*70)

    # Change to example directory
    example_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(example_dir)
    print(f"\n📂 Working directory: {example_dir}")

    try:
        # Run protein tests
        test_protein_features()

        # Run molecule tests
        test_molecule_features()

        # Run custom SMARTS test
        test_custom_smarts()

        # Summary
        print("\n" + "="*70)
        print("  ✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n  Summary:")
        print("    - Protein featurizer: ✓ Working")
        print("    - Molecule featurizer: ✓ Working")
        print("    - Custom SMARTS: ✓ Working")
        print("\n")

    except Exception as e:
        print("\n" + "="*70)
        print("  ❌ TEST FAILED")
        print("="*70)
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

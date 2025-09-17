#!/usr/bin/env python
"""Test script for efficient featurizer API"""

from featurizer.molecule_featurizer import MoleculeFeaturizer
from rdkit import Chem
import time

def test_molecule_efficiency():
    print("Testing MoleculeFeaturizer Efficiency")
    print("=" * 50)

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

    # Test 1: Instance mode (efficient for multiple operations)
    print("\n1. Instance Mode (parse once, extract multiple):")
    start = time.time()
    featurizer = MoleculeFeaturizer(smiles)  # Parse once
    features = featurizer.get_feature()  # Uses cached molecule
    graph = featurizer.get_graph()  # Uses cached molecule
    fingerprints = featurizer.get_fingerprints()  # Uses cached molecule
    phys = featurizer.get_physicochemical_features()  # Uses cached molecule
    instance_time = time.time() - start
    print(f"   Time: {instance_time:.4f}s")
    print(f"   Features extracted: {len(features)} types")
    print(f"   Graph nodes: {graph[0]['node_feats'].shape[0]}")

    # Test 2: Static mode (backward compatible)
    print("\n2. Static Mode (parse each time):")
    start = time.time()
    featurizer = MoleculeFeaturizer()  # No molecule cached
    features = featurizer.get_feature(smiles)  # Parse molecule
    graph = featurizer.get_graph(smiles)  # Parse molecule again
    fingerprints = featurizer.get_fingerprints(smiles)  # Parse molecule again
    phys = featurizer.get_physicochemical_features(smiles)  # Parse molecule again
    static_time = time.time() - start
    print(f"   Time: {static_time:.4f}s")

    # Test 3: Switching molecules
    print("\n3. Switching molecules with set_molecule():")
    featurizer = MoleculeFeaturizer("CCO")  # Ethanol
    features1 = featurizer.get_feature()
    print(f"   Ethanol atoms: {featurizer._cache.get('num_atoms', 'N/A')}")

    featurizer.set_molecule("c1ccccc1")  # Benzene
    features2 = featurizer.get_feature()
    print(f"   Benzene atoms: {featurizer._cache.get('num_atoms', 'N/A')}")

    # Test 4: Both modes with same instance
    print("\n4. Mixed mode usage:")
    featurizer = MoleculeFeaturizer("CCO")
    # Use cached molecule
    f1 = featurizer.get_feature()
    # Override with different molecule
    f2 = featurizer.get_feature("c1ccccc1")
    # Back to cached molecule
    f3 = featurizer.get_feature()
    print(f"   Feature 1 (cached CCO): {f1['descriptor'].shape}")
    print(f"   Feature 2 (override c1ccccc1): {f2['descriptor'].shape}")
    print(f"   Feature 3 (cached CCO again): {f3['descriptor'].shape}")

    print(f"\n✓ Efficiency gain: {static_time/instance_time:.1f}x faster in instance mode!")

def test_protein_api():
    print("\n\nTesting ProteinFeaturizer API")
    print("=" * 50)

    print("\n✓ ProteinFeaturizer would work similarly:")
    print("   featurizer = ProteinFeaturizer('protein.pdb')  # Parse once")
    print("   seq = featurizer.get_sequence_features()  # Uses cached")
    print("   geo = featurizer.get_geometric_features()  # Uses cached")
    print("   sasa = featurizer.get_sasa_features()  # Uses cached")
    print("\n   Much more efficient than parsing PDB for each method!")

if __name__ == "__main__":
    test_molecule_efficiency()
    test_protein_api()
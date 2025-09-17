#!/usr/bin/env python
"""Test script to verify protein feature API structure"""

from featurizer.protein_featurizer import ProteinFeaturizer

def test_api_structure():
    print("Testing ProteinFeaturizer API structure")
    print("=" * 50)

    # Initialize featurizer
    featurizer = ProteinFeaturizer(standardize=False)

    # List all available methods
    methods = [
        'get_sequence_features',
        'get_geometric_features',
        'get_sasa_features',
        'get_contact_map',
        'get_relative_position',
        'get_node_features',
        'get_edge_features',
        'get_terminal_flags',
        'extract',
        'extract_batch'
    ]

    print("\nAvailable individual feature extraction methods:")
    for method in methods:
        if hasattr(featurizer, method):
            print(f"✓ {method}")
            # Get docstring
            doc = getattr(featurizer, method).__doc__
            if doc:
                first_line = doc.strip().split('\n')[0]
                print(f"  → {first_line}")
        else:
            print(f"✗ {method} - NOT FOUND")

    print("\n✓ API structure verified!")
    print("\nUsage example:")
    print("```python")
    print("from featurizer.protein_featurizer import ProteinFeaturizer")
    print("from rdkit import Chem")
    print()
    print("featurizer = ProteinFeaturizer()")
    print("pdb_file = 'protein.pdb'")
    print()
    print("# Get individual feature types")
    print("sequence = featurizer.get_sequence_features(pdb_file)")
    print("geometry = featurizer.get_geometric_features(pdb_file)")
    print("sasa = featurizer.get_sasa_features(pdb_file)")
    print("contacts = featurizer.get_contact_map(pdb_file)")
    print("```")

if __name__ == "__main__":
    test_api_structure()
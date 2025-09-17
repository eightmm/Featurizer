#!/usr/bin/env python
"""Test script for individual protein feature extraction methods"""

import os
import torch
from featurizer.protein_featurizer import ProteinFeaturizer

def test_individual_features():
    # Find a test PDB file
    pdb_files = []
    for root, dirs, files in os.walk('/home/jaemin/git/Featurizer'):
        for file in files:
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))

    if not pdb_files:
        print("No PDB files found in repository")
        return

    pdb_file = pdb_files[0]
    print(f"Testing with PDB file: {pdb_file}")
    print("=" * 50)

    # Initialize featurizer
    featurizer = ProteinFeaturizer(standardize=False)  # Assuming file is already clean

    # Test sequence features
    print("\n1. Testing get_sequence_features():")
    try:
        seq_features = featurizer.get_sequence_features(pdb_file)
        print(f"   - Residue types shape: {seq_features['residue_types'].shape}")
        print(f"   - One-hot encoding shape: {seq_features['residue_one_hot'].shape}")
        print(f"   - Number of residues: {seq_features['num_residues']}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test geometric features
    print("\n2. Testing get_geometric_features():")
    try:
        geo_features = featurizer.get_geometric_features(pdb_file)
        print(f"   - Dihedrals shape: {geo_features['dihedrals'].shape}")
        print(f"   - Has chi angles shape: {geo_features['has_chi_angles'].shape}")
        print(f"   - Backbone curvature shape: {geo_features['backbone_curvature'].shape}")
        print(f"   - Backbone torsion shape: {geo_features['backbone_torsion'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test SASA features
    print("\n3. Testing get_sasa_features():")
    try:
        sasa = featurizer.get_sasa_features(pdb_file)
        print(f"   - SASA tensor shape: {sasa.shape}")
        print(f"   - SASA features per residue: {sasa.shape[1] if sasa.dim() > 1 else 'N/A'}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test contact map
    print("\n4. Testing get_contact_map():")
    try:
        contact_map = featurizer.get_contact_map(pdb_file, cutoff=8.0)
        print(f"   - Adjacency matrix shape: {contact_map['adjacency_matrix'].shape}")
        print(f"   - Number of contacts: {contact_map['adjacency_matrix'].sum().item()}")
        src, dst = contact_map['edges']
        print(f"   - Number of edges: {len(src)}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test relative position
    print("\n5. Testing get_relative_position():")
    try:
        rel_pos = featurizer.get_relative_position(pdb_file, cutoff=32)
        print(f"   - Relative position shape: {rel_pos.shape}")
        print(f"   - Encoding dimensions: {rel_pos.shape[-1] if rel_pos.dim() > 2 else 'N/A'}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test node features
    print("\n6. Testing get_node_features():")
    try:
        node_features = featurizer.get_node_features(pdb_file)
        print(f"   - Coordinates shape: {node_features['coordinates'].shape}")
        print(f"   - Number of scalar features: {len(node_features['scalar_features'])}")
        print(f"   - Number of vector features: {len(node_features['vector_features'])}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test edge features
    print("\n7. Testing get_edge_features():")
    try:
        edge_features = featurizer.get_edge_features(pdb_file, distance_cutoff=8.0)
        src, dst = edge_features['edges']
        print(f"   - Number of edges: {len(src)}")
        print(f"   - Number of scalar edge features: {len(edge_features['scalar_features'])}")
        print(f"   - Number of vector edge features: {len(edge_features['vector_features'])}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test terminal flags
    print("\n8. Testing get_terminal_flags():")
    try:
        terminal = featurizer.get_terminal_flags(pdb_file)
        print(f"   - N-terminal flags shape: {terminal['n_terminal'].shape}")
        print(f"   - C-terminal flags shape: {terminal['c_terminal'].shape}")
        print(f"   - N-terminal residues: {terminal['n_terminal'].sum().item()}")
        print(f"   - C-terminal residues: {terminal['c_terminal'].sum().item()}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_individual_features()
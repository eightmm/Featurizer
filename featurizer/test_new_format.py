#!/usr/bin/env python
"""Test script for the new get_graph format"""

import torch
from rdkit import Chem
from molecule_featurizer.molecule_feature import MoleculeFeaturizer

def test_new_graph_format():
    featurizer = MoleculeFeaturizer()

    # Test molecule (aspirin)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)

    print("Testing new get_graph format")
    print("=" * 50)

    # Test with SMILES
    print("\n1. Testing with SMILES:")
    node, edge = featurizer.get_graph(smiles)

    print(f"   Node dictionary keys: {list(node.keys())}")
    print(f"   - coords shape: {node['coords'].shape}")
    print(f"   - node_feats shape: {node['node_feats'].shape}")

    print(f"\n   Edge dictionary keys: {list(edge.keys())}")
    print(f"   - edges shape: {edge['edges'].shape}")
    print(f"   - edge_feats shape: {edge['edge_feats'].shape}")
    print(f"   - Number of edges: {edge['edges'].shape[1]}")

    # Test with RDKit mol
    print("\n2. Testing with RDKit mol:")
    node, edge = featurizer.get_graph(mol)

    print(f"   Node dictionary keys: {list(node.keys())}")
    print(f"   - coords shape: {node['coords'].shape}")
    print(f"   - node_feats shape: {node['node_feats'].shape}")

    print(f"\n   Edge dictionary keys: {list(edge.keys())}")
    print(f"   - edges shape: {edge['edges'].shape}")
    print(f"   - edge_feats shape: {edge['edge_feats'].shape}")

    # Verify edge format
    print("\n3. Verifying edge format:")
    src_indices = edge['edges'][0]  # First row is source indices
    dst_indices = edge['edges'][1]  # Second row is destination indices
    print(f"   - Source indices range: [{src_indices.min()}, {src_indices.max()}]")
    print(f"   - Destination indices range: [{dst_indices.min()}, {dst_indices.max()}]")
    print(f"   - Edge features per edge: {edge['edge_feats'].shape[1]}")

    # Test unpacking
    print("\n4. Testing tuple unpacking:")
    n, e = featurizer.get_graph(mol)
    print(f"   - Successfully unpacked as: node, edge")
    print(f"   - Node has coords: {'coords' in n}")
    print(f"   - Node has node_feats: {'node_feats' in n}")
    print(f"   - Edge has edges: {'edges' in e}")
    print(f"   - Edge has edge_feats: {'edge_feats' in e}")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_new_graph_format()
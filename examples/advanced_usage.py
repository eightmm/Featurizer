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
    MolecularFeatureExtractor,
    MolecularGraph,
    create_molecular_graph,
    smiles_to_graph,
    ProteinFeaturizer,
    PDBStandardizer,
    ResidueFeaturizer
)


def component_usage_example():
    """
    Using individual components separately.
    """
    print("Using Individual Components")
    print("=" * 50)

    print("\nStep-by-step feature extraction:")
    print("-" * 40)
    print("""
    from protein_featurizer import PDBStandardizer, ResidueFeaturizer

    # Step 1: Standardize PDB
    standardizer = PDBStandardizer(remove_hydrogens=True)
    clean_pdb = standardizer.standardize('input.pdb', 'clean.pdb')

    # Step 2: Extract features
    featurizer = ResidueFeaturizer(clean_pdb)

    # Step 3: Get specific features
    residues = featurizer.get_residues()
    sasa = featurizer.calculate_sasa()
    terminal_flags = featurizer.get_terminal_flags()

    # Step 4: Get all features
    node_features, edge_features = featurizer.get_features()
    """)


def custom_pipeline_example():
    """
    Creating a custom processing pipeline.
    """
    print("\n" + "=" * 50)
    print("Custom Processing Pipeline")
    print("=" * 50)

    print("\nBuilding a custom pipeline:")
    print("-" * 40)
    print("""
    import torch
    from protein_featurizer import Featurizer

    class CustomPipeline:
        def __init__(self):
            self.featurizer = Featurizer()

        def process_with_filtering(self, pdb_file):
            # Extract features
            features = self.featurizer.extract(pdb_file)

            # Custom filtering - only keep proteins with > 50 residues
            num_residues = len(features['node']['coord'])
            if num_residues < 50:
                return None

            # Add custom metadata
            features['custom_metadata'] = {
                'processed_by': 'CustomPipeline',
                'num_residues': num_residues,
                'passed_filter': True
            }

            return features

        def process_and_normalize(self, pdb_file):
            features = self.process_with_filtering(pdb_file)

            if features:
                # Normalize coordinates
                coords = features['node']['coord']
                coords_normalized = (coords - coords.mean(0)) / coords.std(0)
                features['node']['coord_normalized'] = coords_normalized

            return features

    # Use custom pipeline
    pipeline = CustomPipeline()
    features = pipeline.process_and_normalize('protein.pdb')
    """)


def feature_analysis_example():
    """
    Analyzing extracted features.
    """
    print("\n" + "=" * 50)
    print("Feature Analysis Example")
    print("=" * 50)

    print("\nAnalyzing extracted features:")
    print("-" * 40)
    print("""
    import numpy as np
    import torch
    from protein_featurizer import Featurizer

    featurizer = Featurizer()
    features = featurizer.extract('protein.pdb')

    # Analyze node features
    node_features = features['node']
    coords = node_features['coord']

    # Calculate statistics
    num_residues = len(coords)
    center_of_mass = coords.mean(dim=0)
    radius_of_gyration = torch.sqrt(((coords - center_of_mass) ** 2).sum() / num_residues)

    print(f"Number of residues: {num_residues}")
    print(f"Center of mass: {center_of_mass}")
    print(f"Radius of gyration: {radius_of_gyration:.2f} Å")

    # Analyze edge features
    edge_features = features['edge']
    src, dst = edge_features['edges']
    num_edges = len(src)
    avg_degree = num_edges / num_residues

    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")

    # Analyze scalar features
    scalar_features = node_features['node_scalar_features']
    residue_types = scalar_features[0]  # One-hot encoded residue types

    # Get residue composition
    residue_counts = residue_types.sum(dim=0)
    most_common_idx = residue_counts.argmax()
    print(f"Most common residue type index: {most_common_idx}")
    """)


def integration_example():
    """
    Integration with machine learning frameworks.
    """
    print("\n" + "=" * 50)
    print("ML Integration Example")
    print("=" * 50)

    print("\nIntegrating with PyTorch for ML:")
    print("-" * 40)
    print("""
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from protein_featurizer import Featurizer

    def create_graph_data(pdb_file):
        # Extract features
        featurizer = Featurizer()
        features = featurizer.extract(pdb_file)

        # Create PyTorch Geometric data object
        node_features = features['node']
        edge_features = features['edge']

        # Prepare node features
        coords = node_features['coord']
        scalar_features = torch.cat(node_features['node_scalar_features'], dim=1)

        # Prepare edge features
        edge_index = torch.stack(edge_features['edges'])
        edge_attr = torch.cat(edge_features['edge_scalar_features'], dim=1)

        # Create graph data
        data = Data(
            x=scalar_features,
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return data

    # Use in training
    graph_data = create_graph_data('protein.pdb')
    print(f"Graph nodes: {graph_data.num_nodes}")
    print(f"Graph edges: {graph_data.num_edges}")
    print(f"Node features: {graph_data.x.shape}")
    print(f"Edge features: {graph_data.edge_attr.shape}")
    """)


def memory_efficient_example():
    """
    Memory-efficient processing for large datasets.
    """
    print("\n" + "=" * 50)
    print("Memory-Efficient Processing")
    print("=" * 50)

    print("\nProcessing large datasets efficiently:")
    print("-" * 40)
    print("""
    from pathlib import Path
    import torch
    from protein_featurizer import Featurizer

    def process_large_dataset(pdb_dir, output_dir, batch_size=10):
        pdb_dir = Path(pdb_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Get all PDB files
        all_pdbs = list(pdb_dir.glob('*.pdb'))

        # Process in batches to manage memory
        featurizer = Featurizer()

        for i in range(0, len(all_pdbs), batch_size):
            batch = all_pdbs[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{len(all_pdbs)//batch_size + 1}")

            for pdb_file in batch:
                try:
                    output_file = output_dir / f"{pdb_file.stem}_features.pt"

                    # Extract and immediately save to disk
                    features = featurizer.extract(str(pdb_file))
                    torch.save(features, output_file)

                    # Clear from memory
                    del features

                except Exception as e:
                    print(f"Failed: {pdb_file.name} - {e}")
                    continue

            # Optional: Force garbage collection after each batch
            import gc
            gc.collect()

    # Run processing
    process_large_dataset('data/pdbs', 'data/features', batch_size=20)
    """)


if __name__ == "__main__":
    component_usage_example()
    custom_pipeline_example()
    feature_analysis_example()
    integration_example()
    memory_efficient_example()

    print("\n" + "=" * 50)
    print("These examples demonstrate advanced usage patterns.")
    print("Adapt them to your specific needs.")
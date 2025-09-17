#!/usr/bin/env python3
"""
Basic usage example of the Featurizer package.
"""

from featurizer import (
    ProteinFeaturizer,
    create_molecule_features,
    extract_molecule_features,
    extract_protein_features
)
from rdkit import Chem


def molecule_features_example():
    """
    Demonstrate molecule feature extraction from SMILES and mol objects.
    """
    print("Molecule Features Example")
    print("=" * 50)

    # Example molecule: Aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    print(f"\nExtracting features from SMILES: {smiles}")

    # Method 1: From SMILES string
    features = create_molecule_features(smiles)

    print("✓ Features extracted successfully!")
    print(f"  - Descriptor dimensions: {features['descriptor'].shape}")
    print(f"  - Morgan fingerprint size: {features['morgan'].shape}")
    print(f"  - MACCS keys size: {features['maccs'].shape}")

    # Method 2: From RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    features2 = create_molecule_features(mol, add_hs=False)

    print("\n✓ Features from mol object extracted!")
    print(f"  - RDKit fingerprint size: {features2['rdkit'].shape}")
    print(f"  - Pharmacophore fingerprint size: {features2['pharmacophore2d'].shape}")


def protein_features_example():
    """
    Demonstrate protein feature extraction from a PDB file.
    """
    print("\n" + "=" * 50)
    print("Protein Features Example")
    print("=" * 50)

    # Initialize featurizer with default settings
    featurizer = ProteinFeaturizer()

    # Extract features from a PDB file
    # Replace 'protein.pdb' with your actual PDB file path
    pdb_file = "protein.pdb"

    try:
        print(f"\nExtracting features from {pdb_file}...")
        features = featurizer.extract(pdb_file)

        # Access the extracted features
        node_features = features['node']
        edge_features = features['edge']
        metadata = features['metadata']

        print(f"✓ Features extracted successfully!")
        print(f"  - Number of residues: {len(node_features['coord'])}")
        print(f"  - Number of edges: {len(edge_features['edges'][0])}")
        print(f"  - Standardized: {metadata['standardized']}")

    except FileNotFoundError:
        print(f"⚠ File '{pdb_file}' not found.")
        print("  Please provide a valid PDB file path.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def convenience_functions_example():
    """
    Demonstrate using convenience functions for quick feature extraction.
    """
    print("\n" + "=" * 50)
    print("Convenience Functions Example")
    print("=" * 50)

    # Quick molecule feature extraction
    smiles = "CCO"  # Ethanol
    mol_features = extract_molecule_features(smiles)
    print(f"\n✓ Molecule features for {smiles}:")
    print(f"  - Number of feature types: {len(mol_features)}")

    # Quick protein feature extraction (if file exists)
    pdb_file = "protein.pdb"
    try:
        protein_features = extract_protein_features(pdb_file, standardize=True)
        print(f"\n✓ Protein features extracted from {pdb_file}")
    except:
        print(f"\n⚠ Skipping protein example (file not found)")


def save_features_example():
    """
    Demonstrate saving features to files.
    """
    print("\n" + "=" * 50)
    print("Saving Features Example")
    print("=" * 50)

    # Save molecule features
    smiles = "CC(C)CC1=CC=C(C=C1)C(C)C"  # Ibuprofen
    features = create_molecule_features(smiles)

    try:
        import torch
        torch.save(features, "ibuprofen_features.pt")
        print("✓ Molecule features saved to 'ibuprofen_features.pt'")
    except ImportError:
        print("⚠ PyTorch not available for saving")

    # Save protein features (if file exists)
    try:
        featurizer = ProteinFeaturizer()
        features = featurizer.extract("protein.pdb", save_to="protein_features.pt")
        print("✓ Protein features saved to 'protein_features.pt'")
    except:
        print("⚠ Skipping protein save (file not found)")


def custom_options_example():
    """
    Demonstrate using custom options for feature extraction.
    """
    print("\n" + "=" * 50)
    print("Custom Options Example")
    print("=" * 50)

    # Molecule features without adding hydrogens
    smiles = "c1ccccc1"  # Benzene
    features = create_molecule_features(smiles, add_hs=False)
    print(f"✓ Extracted features without adding hydrogens")

    # Protein features without standardization
    featurizer = ProteinFeaturizer(standardize=False)
    print("✓ Created protein featurizer without standardization step")

    # Protein features keeping hydrogens
    featurizer_with_h = ProteinFeaturizer(keep_hydrogens=True)
    print("✓ Created protein featurizer that keeps hydrogen atoms")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Featurizer Package - Basic Usage Examples")
    print("=" * 60)

    # Run examples
    molecule_features_example()
    protein_features_example()
    convenience_functions_example()
    save_features_example()
    custom_options_example()

    print("\n" + "=" * 60)
    print(" Examples completed!")
    print("=" * 60)
#!/usr/bin/env python3
"""
Basic usage example of the Protein Featurizer package.
"""

from protein_featurizer import Featurizer


def basic_example():
    """
    Demonstrate basic feature extraction from a single PDB file.
    """
    print("Basic Usage Example")
    print("=" * 50)

    # Initialize featurizer with default settings
    featurizer = Featurizer()

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


def save_features_example():
    """
    Demonstrate saving features to a file.
    """
    print("\n" + "=" * 50)
    print("Saving Features Example")
    print("=" * 50)

    featurizer = Featurizer()

    # Extract and save features
    pdb_file = "protein.pdb"
    output_file = "protein_features.pt"

    try:
        print(f"\nProcessing and saving to {output_file}...")
        features = featurizer.extract(pdb_file, save_to=output_file)
        print(f"✓ Features saved to {output_file}")

    except FileNotFoundError:
        print("⚠ Example code - replace with your PDB file")
        print(f"  featurizer.extract('your_protein.pdb', save_to='{output_file}')")


def no_standardization_example():
    """
    Demonstrate feature extraction without PDB standardization.
    """
    print("\n" + "=" * 50)
    print("Using Pre-cleaned PDB Example")
    print("=" * 50)

    # Skip standardization for already clean PDB files
    featurizer = Featurizer(standardize=False)

    print("\nExample code:")
    print("  # For already clean/standardized PDB files")
    print("  featurizer = Featurizer(standardize=False)")
    print("  features = featurizer.extract('clean_protein.pdb')")
    print("\n  # Or use class method")
    print("  features = Featurizer.from_clean_pdb('clean_protein.pdb')")


if __name__ == "__main__":
    # Run examples
    basic_example()
    save_features_example()
    no_standardization_example()

    print("\n" + "=" * 50)
    print("For more examples, see other files in the examples/ directory")
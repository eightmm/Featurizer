#!/usr/bin/env python3
"""
Example usage of the Protein Featurizer package.
"""

from protein_featurizer import Featurizer


def main():
    """
    Demonstrate basic usage of the Protein Featurizer.
    """
    print("Protein Featurizer Example")
    print("=" * 40)

    # Initialize featurizer
    print("\n1. Initializing featurizer...")
    featurizer = Featurizer(standardize=True, keep_hydrogens=False)

    # Example: Single file processing
    print("\n2. Processing single PDB file...")
    try:
        # Replace with your PDB file path
        pdb_file = "example.pdb"
        features = featurizer.extract(pdb_file, save_to="features.pt")

        print(f"   ✓ Features extracted successfully!")
        print(f"   - Number of residues: {len(features['node']['coord'])}")
        print(f"   - Number of edges: {len(features['edge']['edges'][0])}")

    except FileNotFoundError:
        print("   ⚠ Example PDB file not found. Please provide a valid PDB file.")
        print("   Usage: features = featurizer.extract('your_protein.pdb')")

    # Example: Batch processing
    print("\n3. Batch processing example:")
    print("   ```python")
    print("   pdb_files = ['protein1.pdb', 'protein2.pdb', 'protein3.pdb']")
    print("   results = featurizer.extract_batch(pdb_files, output_dir='features/')")
    print("   ```")

    # Example: Using clean PDB
    print("\n4. Using pre-cleaned PDB:")
    print("   ```python")
    print("   features = Featurizer.from_clean_pdb('clean_protein.pdb')")
    print("   ```")

    # Example: Standardize only
    print("\n5. Standardize PDB only:")
    print("   ```python")
    print("   Featurizer.standardize_only('input.pdb', 'clean.pdb')")
    print("   ```")

    print("\n" + "=" * 40)
    print("For more examples, see the README.md file")


if __name__ == "__main__":
    main()
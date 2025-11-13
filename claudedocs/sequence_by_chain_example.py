#!/usr/bin/env python3
"""
Example script demonstrating chain-based sequence extraction functionality.

This script shows how to use:
- get_sequence_by_chain() - Get sequences separated by chain ID

Usage:
    python sequence_by_chain_example.py <path_to_pdb_file>
"""

import sys
from featurizer.protein_featurizer import ProteinFeaturizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python sequence_by_chain_example.py <path_to_pdb_file>")
        print("\nExample:")
        print("  python sequence_by_chain_example.py protein.pdb")
        sys.exit(1)

    pdb_file = sys.argv[1]

    # Initialize the featurizer
    print(f"Loading PDB file: {pdb_file}")
    featurizer = ProteinFeaturizer(pdb_file, standardize=True)

    # Get sequences separated by chain
    print("\n" + "="*60)
    print("Sequences by Chain:")
    print("="*60)
    sequences_by_chain = featurizer.get_sequence_by_chain()

    for chain_id, sequence in sequences_by_chain.items():
        print(f"\nChain {chain_id}:")
        print(f"  Length: {len(sequence)}")
        print(f"  Sequence: {sequence}")

    # Get full sequence by concatenating if needed
    full_sequence = ''.join(sequences_by_chain.values())
    print("\n" + "="*60)
    print("Full Sequence (all chains concatenated):")
    print("="*60)
    print(full_sequence)
    print(f"\nTotal residues: {len(full_sequence)}")

    # Additional info: Show chain composition
    print("\n" + "="*60)
    print("Chain Composition:")
    print("="*60)
    total_chains = len(sequences_by_chain)
    print(f"Total chains: {total_chains}")

    for chain_id, sequence in sequences_by_chain.items():
        # Count amino acid types
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        print(f"\nChain {chain_id}:")
        print(f"  Residues: {len(sequence)}")
        print(f"  Unique amino acids: {len(aa_counts)}")
        print(f"  Composition: {', '.join(f'{aa}:{count}' for aa, count in sorted(aa_counts.items()))}")


if __name__ == "__main__":
    main()

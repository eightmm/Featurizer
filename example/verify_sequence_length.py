#!/usr/bin/env python3
"""
Verify that sequence lengths match structure residue counts.

This script confirms that:
1. Sum of chain sequence lengths = total residue count in structure
2. Each chain's sequence length matches its residue count
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from featurizer import ProteinFeaturizer


def verify_sequence_lengths(pdb_file):
    """Verify sequence and structure lengths match."""
    print(f"Analyzing: {pdb_file}\n")

    featurizer = ProteinFeaturizer(pdb_file, standardize=True)

    # Get sequences by chain
    sequences = featurizer.get_sequence_by_chain()

    # Get structure info
    residues = featurizer._featurizer.get_residues()
    num_residues = featurizer.num_residues

    print("="*60)
    print("SEQUENCE LENGTHS")
    print("="*60)

    total_seq_length = 0
    chain_residue_counts = {}

    # Count residues per chain from structure
    for chain, res_num, res_type in residues:
        chain_residue_counts[chain] = chain_residue_counts.get(chain, 0) + 1

    # Compare sequence lengths with structure
    all_match = True
    for chain_id, sequence in sequences.items():
        seq_length = len(sequence)
        struct_count = chain_residue_counts.get(chain_id, 0)
        match = "✓" if seq_length == struct_count else "✗"

        print(f"\nChain {chain_id}:")
        print(f"  Sequence length:  {seq_length}")
        print(f"  Structure count:  {struct_count}")
        print(f"  Match: {match}")

        if seq_length != struct_count:
            all_match = False

        total_seq_length += seq_length

    print("\n" + "="*60)
    print("TOTAL COUNTS")
    print("="*60)
    print(f"Sum of sequence lengths: {total_seq_length}")
    print(f"Structure residue count: {num_residues}")
    print(f"Match: {'✓' if total_seq_length == num_residues else '✗'}")

    print("\n" + "="*60)
    if all_match and total_seq_length == num_residues:
        print("✓ VERIFICATION PASSED: All lengths match!")
        print("="*60)
        return True
    else:
        print("✗ VERIFICATION FAILED: Length mismatch detected!")
        print("="*60)
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
    else:
        pdb_file = "10gs_protein.pdb"

    if not os.path.exists(pdb_file):
        print(f"Error: File not found: {pdb_file}")
        sys.exit(1)

    success = verify_sequence_lengths(pdb_file)
    sys.exit(0 if success else 1)

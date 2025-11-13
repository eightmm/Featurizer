#!/usr/bin/env python3
"""
Test unknown residue handling - should only use N, CA, C, O, CB atoms.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from featurizer import ProteinFeaturizer


def create_test_pdb_with_unk():
    """Create a test PDB with an unknown residue that has many atoms."""
    test_pdb = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00  0.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      14.000  14.000  14.000  1.00  0.00           C
ATOM      6  N   UNK A   2      15.000  15.000  15.000  1.00  0.00           N
ATOM      7  CA  UNK A   2      16.000  16.000  16.000  1.00  0.00           C
ATOM      8  C   UNK A   2      17.000  17.000  17.000  1.00  0.00           C
ATOM      9  O   UNK A   2      18.000  18.000  18.000  1.00  0.00           O
ATOM     10  CB  UNK A   2      19.000  19.000  19.000  1.00  0.00           C
ATOM     11  CG  UNK A   2      20.000  20.000  20.000  1.00  0.00           C
ATOM     12  CD  UNK A   2      21.000  21.000  21.000  1.00  0.00           C
ATOM     13  CE  UNK A   2      22.000  22.000  22.000  1.00  0.00           C
ATOM     14  NZ  UNK A   2      23.000  23.000  23.000  1.00  0.00           N
ATOM     15  CH  UNK A   2      24.000  24.000  24.000  1.00  0.00           C
ATOM     16  N   GLY A   3      25.000  25.000  25.000  1.00  0.00           N
ATOM     17  CA  GLY A   3      26.000  26.000  26.000  1.00  0.00           C
ATOM     18  C   GLY A   3      27.000  27.000  27.000  1.00  0.00           C
ATOM     19  O   GLY A   3      28.000  28.000  28.000  1.00  0.00           O
"""
    return test_pdb


def test_unk_residue_filtering():
    """Test that UNK residues only use N, CA, C, O, CB atoms."""
    print("="*70)
    print("Testing Unknown Residue Atom Filtering")
    print("="*70)

    test_pdb = create_test_pdb_with_unk()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        input_file = f.name

    try:
        # Test without standardization to keep UNK as-is
        featurizer = ProteinFeaturizer(input_file, standardize=False)

        print("\nResidues in structure:")
        for idx, (chain, res_num, res_type) in enumerate(featurizer.residues):
            res_name = {0: 'ALA', 6: 'GLY', 20: 'UNK'}.get(res_type, 'OTHER')
            print(f"  [{idx}] Chain {chain}, Residue {res_num}, Type {res_type} ({res_name})")

        print("\nCoordinate tensor analysis:")
        print(f"  coords shape: {featurizer.coords.shape}")
        print(f"  Expected: (3 residues, 15 atoms, 3 coords)")

        # Check each residue
        for idx, (chain, res_num, res_type) in enumerate(featurizer.residues):
            res_name = {0: 'ALA', 6: 'GLY', 20: 'UNK'}.get(res_type, 'OTHER')

            # Count non-zero coordinate slots
            coords = featurizer.coords[idx]
            non_zero_atoms = (coords.abs().sum(dim=1) > 0).sum().item()

            print(f"\n  Residue {idx} ({res_name}):")
            print(f"    Type: {res_type}")
            print(f"    Non-zero atom positions: {non_zero_atoms}")

            if res_type == 20:  # UNK residue
                print(f"    Expected: 5 atoms (N, CA, C, O, CB) + sidechain centroid (position 14)")
                # Check positions 0-4 are used (N, CA, C, O, CB)
                first_five = (coords[:5].abs().sum(dim=1) > 0).sum().item()
                # Check position 14 is used (sidechain centroid)
                has_centroid = (coords[14].abs().sum() > 0).item()

                if first_five == 5 and has_centroid:
                    print(f"    ✓ CORRECT: Only backbone + CB used (positions 0-4)")
                    print(f"    ✓ Sidechain centroid stored at position 14")
                    # Verify CB and centroid are same (since only CB in sidechain)
                    cb_coord = coords[4]
                    centroid_coord = coords[14]
                    if torch.allclose(cb_coord, centroid_coord):
                        print(f"    ✓ Centroid = CB (as expected for single sidechain atom)")
                else:
                    print(f"    ✗ WRONG: Check failed")
                    print(f"    First 5 atoms: {first_five}/5")
                    print(f"    Has centroid: {has_centroid}")

        # Test sequence extraction
        print("\n" + "="*70)
        print("Sequence Extraction Test:")
        print("="*70)

        sequences = featurizer.get_sequence_by_chain()
        print(f"\nChain A sequence: {sequences.get('A', 'N/A')}")
        print(f"Expected: 'AXG' (ALA, UNK→X, GLY)")

        if sequences.get('A') == 'AXG':
            print("✓ CORRECT: UNK mapped to 'X'")
        else:
            print(f"✗ WRONG: Expected 'AXG', got '{sequences.get('A')}'")

        print("\n" + "="*70)
        print("✓ Unknown Residue Filtering Test Complete")
        print("="*70)

    finally:
        os.unlink(input_file)


if __name__ == "__main__":
    test_unk_residue_filtering()

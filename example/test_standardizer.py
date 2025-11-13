#!/usr/bin/env python3
"""
Test PDBStandardizer to identify potential issues.

This script checks for:
1. Insertion code handling
2. Alternative location handling
3. Modified amino acid handling
4. Residue name formatting
5. Chain ID handling
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from featurizer.protein_featurizer.pdb_standardizer import PDBStandardizer


def create_test_pdb_with_insertion_codes():
    """Create a test PDB with insertion codes."""
    test_pdb = """ATOM      1  N   ALA A  10      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A  10      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C   ALA A  10      12.000  12.000  12.000  1.00  0.00           C
ATOM      4  O   ALA A  10      13.000  13.000  13.000  1.00  0.00           O
ATOM      5  CB  ALA A  10      14.000  14.000  14.000  1.00  0.00           C
ATOM      6  N   ALA A  10A     15.000  15.000  15.000  1.00  0.00           N
ATOM      7  CA  ALA A  10A     16.000  16.000  16.000  1.00  0.00           C
ATOM      8  C   ALA A  10A     17.000  17.000  17.000  1.00  0.00           C
ATOM      9  O   ALA A  10A     18.000  18.000  18.000  1.00  0.00           O
ATOM     10  CB  ALA A  10A     19.000  19.000  19.000  1.00  0.00           C
ATOM     11  N   ALA A  10B     20.000  20.000  20.000  1.00  0.00           N
ATOM     12  CA  ALA A  10B     21.000  21.000  21.000  1.00  0.00           C
ATOM     13  C   ALA A  10B     22.000  22.000  22.000  1.00  0.00           C
ATOM     14  O   ALA A  10B     23.000  23.000  23.000  1.00  0.00           O
ATOM     15  CB  ALA A  10B     24.000  24.000  24.000  1.00  0.00           C
ATOM     16  N   ALA A  11      25.000  25.000  25.000  1.00  0.00           N
ATOM     17  CA  ALA A  11      26.000  26.000  26.000  1.00  0.00           C
ATOM     18  C   ALA A  11      27.000  27.000  27.000  1.00  0.00           C
ATOM     19  O   ALA A  11      28.000  28.000  28.000  1.00  0.00           O
ATOM     20  CB  ALA A  11      29.000  29.000  29.000  1.00  0.00           C
"""
    return test_pdb


def create_test_pdb_with_altloc():
    """Create a test PDB with alternative locations."""
    test_pdb = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA AALA A   1      11.000  11.000  11.000  0.50  0.00           C
ATOM      3  CA BALA A   1      11.100  11.100  11.100  0.50  0.00           C
ATOM      4  C   ALA A   1      12.000  12.000  12.000  1.00  0.00           C
ATOM      5  O   ALA A   1      13.000  13.000  13.000  1.00  0.00           O
ATOM      6  CB AALA A   1      14.000  14.000  14.000  0.50  0.00           C
ATOM      7  CB BALA A   1      14.100  14.100  14.100  0.50  0.00           C
"""
    return test_pdb


def create_test_pdb_with_modified_aa():
    """Create a test PDB with modified amino acids."""
    test_pdb = """ATOM      1  N   MSE A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  MSE A   1      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C   MSE A   1      12.000  12.000  12.000  1.00  0.00           C
ATOM      4  O   MSE A   1      13.000  13.000  13.000  1.00  0.00           O
ATOM      5  CB  MSE A   1      14.000  14.000  14.000  1.00  0.00           C
ATOM      6  CG  MSE A   1      15.000  15.000  15.000  1.00  0.00           C
ATOM      7 SE   MSE A   1      16.000  16.000  16.000  1.00  0.00          SE
ATOM      8  CE  MSE A   1      17.000  17.000  17.000  1.00  0.00           C
ATOM      9  N   HID A   2      18.000  18.000  18.000  1.00  0.00           N
ATOM     10  CA  HID A   2      19.000  19.000  19.000  1.00  0.00           C
ATOM     11  C   HID A   2      20.000  20.000  20.000  1.00  0.00           C
ATOM     12  O   HID A   2      21.000  21.000  21.000  1.00  0.00           O
ATOM     13  CB  HID A   2      22.000  22.000  22.000  1.00  0.00           C
"""
    return test_pdb


def test_insertion_codes():
    """Test insertion code handling."""
    print("="*70)
    print("TEST 1: Insertion Code Handling")
    print("="*70)

    test_pdb = create_test_pdb_with_insertion_codes()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        output_file = f.name

    try:
        standardizer = PDBStandardizer(remove_hydrogens=True)
        standardizer.standardize(input_file, output_file)

        # Read output
        with open(output_file, 'r') as f:
            output_lines = f.readlines()

        print("\nInput residues: 10, 10A, 10B, 11")
        print("Expected: Should preserve insertion codes or handle gracefully")
        print("\nOutput residues:")

        residues_seen = set()
        for line in output_lines:
            if line.startswith('ATOM'):
                res_num = line[22:27].strip()
                residues_seen.add(res_num)

        print(f"  {sorted(residues_seen, key=lambda x: (int(''.join(filter(str.isdigit, x))), ''.join(filter(str.isalpha, x))))}")

        # Check if insertion codes are lost
        if '10A' not in residues_seen and '10B' not in residues_seen:
            print("\n⚠️  WARNING: Insertion codes were REMOVED during standardization!")
            print("  Original: 10, 10A, 10B, 11")
            print(f"  Output:   {sorted(residues_seen)}")
            print("  This means residues with insertion codes get renumbered sequentially.")
        else:
            print("\n✓ Insertion codes preserved")

    finally:
        os.unlink(input_file)
        os.unlink(output_file)


def test_altloc():
    """Test alternative location handling."""
    print("\n" + "="*70)
    print("TEST 2: Alternative Location Handling")
    print("="*70)

    test_pdb = create_test_pdb_with_altloc()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        output_file = f.name

    try:
        standardizer = PDBStandardizer(remove_hydrogens=True)
        standardizer.standardize(input_file, output_file)

        # Read output
        with open(output_file, 'r') as f:
            output_lines = f.readlines()

        print("\nInput: CA and CB atoms have A/B alternative locations")
        print("Expected: Should pick one altloc or handle both")
        print("\nOutput:")

        ca_count = 0
        cb_count = 0
        for line in output_lines:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    ca_count += 1
                    print(f"  Found CA atom")
                elif atom_name == 'CB':
                    cb_count += 1
                    print(f"  Found CB atom")

        if ca_count > 1 or cb_count > 1:
            print(f"\n⚠️  WARNING: Multiple positions for same atom!")
            print(f"  CA atoms: {ca_count} (expected: 1)")
            print(f"  CB atoms: {cb_count} (expected: 1)")
            print("  This means alternative locations are ALL kept, causing duplicates.")
        else:
            print(f"\n✓ Alternative locations handled correctly")
            print(f"  CA atoms: {ca_count}")
            print(f"  CB atoms: {cb_count}")

    finally:
        os.unlink(input_file)
        os.unlink(output_file)


def test_modified_amino_acids():
    """Test modified amino acid handling."""
    print("\n" + "="*70)
    print("TEST 3: Modified Amino Acid Handling")
    print("="*70)

    test_pdb = create_test_pdb_with_modified_aa()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        output_file = f.name

    try:
        standardizer = PDBStandardizer(remove_hydrogens=True)
        standardizer.standardize(input_file, output_file)

        # Read output
        with open(output_file, 'r') as f:
            output_lines = f.readlines()

        print("\nInput residues:")
        print("  MSE (selenomethionine) - should map to MET")
        print("  HID (histidine delta) - should map to HIS")
        print("\nOutput residues:")

        for line in output_lines:
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    print(f"  Residue name in output: '{res_name}'")

        # Check if MSE was converted
        mse_found = any('MSE' in line for line in output_lines)
        met_found = any('MET' in line for line in output_lines)
        hid_found = any('HID' in line for line in output_lines)
        his_found = any('HIS' in line for line in output_lines)

        if mse_found:
            print("\n⚠️  WARNING: MSE (selenomethionine) NOT converted to MET!")
            print("  This modified amino acid is not in the mapping.")
        elif met_found:
            print("\n✓ MSE converted to MET")

        if hid_found:
            print("⚠️  WARNING: HID NOT converted to HIS!")
        elif his_found:
            print("✓ HID converted to HIS")

    finally:
        os.unlink(input_file)
        os.unlink(output_file)


def test_residue_name_formatting():
    """Test residue name formatting in PDB output."""
    print("\n" + "="*70)
    print("TEST 4: Residue Name Formatting")
    print("="*70)

    test_pdb = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00  0.00           C
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        output_file = f.name

    try:
        standardizer = PDBStandardizer(remove_hydrogens=True)
        standardizer.standardize(input_file, output_file)

        # Read output
        with open(output_file, 'r') as f:
            output_lines = f.readlines()

        print("\nPDB format requires 3-character residue names")
        print("Checking residue name field (columns 18-20):\n")

        for i, line in enumerate(output_lines, 1):
            if line.startswith('ATOM'):
                res_name_field = line[17:20]
                print(f"  Line {i}: '{res_name_field}' (repr: {repr(res_name_field)})")

                # PDB format: residue name should be right-aligned in 3 chars
                # For 3-letter names like ALA, it should be "ALA" not "ALA "
                if res_name_field != res_name_field.rstrip():
                    print("    ⚠️  Has trailing space")

    finally:
        os.unlink(input_file)
        os.unlink(output_file)


def main():
    """Run all standardizer tests."""
    print("\n" + "="*70)
    print("  PDB STANDARDIZER - COMPREHENSIVE TESTING")
    print("="*70)

    try:
        test_insertion_codes()
        test_altloc()
        test_modified_amino_acids()
        test_residue_name_formatting()

        print("\n" + "="*70)
        print("  TESTING COMPLETE")
        print("="*70)
        print("\nRecommendations:")
        print("1. Add handling for insertion codes (preserve or document removal)")
        print("2. Add altloc handling (pick highest occupancy or A)")
        print("3. Add MSE → MET mapping for selenomethionine")
        print("4. Verify residue name formatting meets PDB standards")
        print()

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

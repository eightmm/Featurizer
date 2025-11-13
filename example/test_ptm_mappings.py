#!/usr/bin/env python3
"""
Test newly added PTM (post-translational modification) mappings.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from featurizer.protein_featurizer.pdb_standardizer import PDBStandardizer


def create_test_pdb_with_ptms():
    """Create a test PDB with various PTMs."""
    test_pdb = """ATOM      1  N   MSE A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  MSE A   1      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C   MSE A   1      12.000  12.000  12.000  1.00  0.00           C
ATOM      4  O   MSE A   1      13.000  13.000  13.000  1.00  0.00           O
ATOM      5  N   SEP A   2      14.000  14.000  14.000  1.00  0.00           N
ATOM      6  CA  SEP A   2      15.000  15.000  15.000  1.00  0.00           C
ATOM      7  C   SEP A   2      16.000  16.000  16.000  1.00  0.00           C
ATOM      8  O   SEP A   2      17.000  17.000  17.000  1.00  0.00           O
ATOM      9  N   TPO A   3      18.000  18.000  18.000  1.00  0.00           N
ATOM     10  CA  TPO A   3      19.000  19.000  19.000  1.00  0.00           C
ATOM     11  C   TPO A   3      20.000  20.000  20.000  1.00  0.00           C
ATOM     12  O   TPO A   3      21.000  21.000  21.000  1.00  0.00           O
ATOM     13  N   PTR A   4      22.000  22.000  22.000  1.00  0.00           N
ATOM     14  CA  PTR A   4      23.000  23.000  23.000  1.00  0.00           C
ATOM     15  C   PTR A   4      24.000  24.000  24.000  1.00  0.00           C
ATOM     16  O   PTR A   4      25.000  25.000  25.000  1.00  0.00           O
ATOM     17  N   HYP A   5      26.000  26.000  26.000  1.00  0.00           N
ATOM     18  CA  HYP A   5      27.000  27.000  27.000  1.00  0.00           C
ATOM     19  C   HYP A   5      28.000  28.000  28.000  1.00  0.00           C
ATOM     20  O   HYP A   5      29.000  29.000  29.000  1.00  0.00           O
ATOM     21  N   MLY A   6      30.000  30.000  30.000  1.00  0.00           N
ATOM     22  CA  MLY A   6      31.000  31.000  31.000  1.00  0.00           C
ATOM     23  C   MLY A   6      32.000  32.000  32.000  1.00  0.00           C
ATOM     24  O   MLY A   6      33.000  33.000  33.000  1.00  0.00           O
ATOM     25  N   CSO A   7      34.000  34.000  34.000  1.00  0.00           N
ATOM     26  CA  CSO A   7      35.000  35.000  35.000  1.00  0.00           C
ATOM     27  C   CSO A   7      36.000  36.000  36.000  1.00  0.00           C
ATOM     28  O   CSO A   7      37.000  37.000  37.000  1.00  0.00           O
"""
    return test_pdb


def test_ptm_mappings():
    """Test PTM residue mappings."""
    print("="*70)
    print("Testing PTM (Post-Translational Modification) Mappings")
    print("="*70)

    test_pdb = create_test_pdb_with_ptms()

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

        # Expected mappings
        expected_mappings = {
            'MSE': 'MET',  # Selenomethionine
            'SEP': 'SER',  # Phosphoserine
            'TPO': 'THR',  # Phosphothreonine
            'PTR': 'TYR',  # Phosphotyrosine
            'HYP': 'PRO',  # Hydroxyproline
            'MLY': 'LYS',  # N-dimethyllysine
            'CSO': 'CYS',  # S-hydroxycysteine
        }

        print("\nInput → Expected → Output:")
        print("-" * 70)

        results = {}
        for line in output_lines:
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                if atom_name == 'CA':  # Just check CA atoms
                    results[res_name] = results.get(res_name, 0) + 1

        all_passed = True
        for original, expected in expected_mappings.items():
            found = expected in results
            status = "✓" if found else "✗"
            print(f"{status}  {original:6s} → {expected:6s} : {'FOUND' if found else 'MISSING'}")
            if not found:
                all_passed = False

        print("\n" + "="*70)
        if all_passed:
            print("✓ ALL PTM MAPPINGS WORKING CORRECTLY!")
        else:
            print("✗ SOME MAPPINGS FAILED!")
        print("="*70)

        # Show what we got
        print("\nActual output residue types found:")
        for res_name, count in sorted(results.items()):
            print(f"  {res_name}: {count} residue(s)")

    finally:
        os.unlink(input_file)
        os.unlink(output_file)


if __name__ == "__main__":
    test_ptm_mappings()

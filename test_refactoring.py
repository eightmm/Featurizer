#!/usr/bin/env python3
"""
Test script to verify the refactored Featurizer package.
"""

import sys
import traceback

def test_imports():
    """Test if all modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)

    tests = []

    # Test molecular featurizer imports
    try:
        from featurizer.molecule_featurizer import MolecularFeatureExtractor
        print("✓ MolecularFeatureExtractor imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ MolecularFeatureExtractor import failed: {e}")
        tests.append(False)

    try:
        from featurizer.molecule_featurizer import create_molecular_features
        print("✓ create_molecular_features imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ create_molecular_features import failed: {e}")
        tests.append(False)

    # Test protein featurizer imports
    try:
        from featurizer.protein_featurizer import PDBStandardizer
        print("✓ PDBStandardizer imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ PDBStandardizer import failed: {e}")
        tests.append(False)

    try:
        from featurizer.protein_featurizer import ResidueFeaturizer
        print("✓ ResidueFeaturizer imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ ResidueFeaturizer import failed: {e}")
        tests.append(False)

    # Test main package imports
    try:
        from featurizer import create_molecular_features
        print("✓ Main package molecular features imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ Main package molecular features import failed: {e}")
        tests.append(False)

    try:
        from featurizer import ProteinFeaturizer
        print("✓ ProteinFeaturizer imported")
        tests.append(True)
    except Exception as e:
        print(f"❌ ProteinFeaturizer import failed: {e}")
        tests.append(False)

    return all(tests)

def test_molecular_features():
    """Test molecular feature extraction functionality."""
    print("\n" + "="*60)
    print("Testing Molecular Feature Extraction")
    print("="*60)

    try:
        # Skip if dependencies not available
        try:
            from rdkit import Chem
            import torch
        except ImportError as e:
            print(f"⚠ Skipping test - missing dependency: {e}")
            return True

        from featurizer.molecule_featurizer import create_molecular_features

        # Test with simple SMILES
        smiles = "CCO"  # Ethanol
        print(f"\nTesting with SMILES: {smiles}")

        # Create features from SMILES
        features = create_molecular_features(smiles, add_hs=False)

        # Check that features were created
        assert 'descriptor' in features, "Missing descriptor features"
        assert features['descriptor'] is not None, "Descriptor is None"

        print(f"✓ Features extracted successfully")
        print(f"  - Descriptor shape: {features['descriptor'].shape}")
        print(f"  - Number of feature types: {len(features)}")

        # Test with RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        features2 = create_molecular_features(mol, add_hs=False)

        assert 'descriptor' in features2, "Missing descriptor from mol object"
        print(f"✓ Features from mol object extracted successfully")

        return True

    except Exception as e:
        print(f"❌ Molecular feature test failed: {e}")
        traceback.print_exc()
        return False

def test_refactoring_changes():
    """Test that refactoring changes are properly applied."""
    print("\n" + "="*60)
    print("Testing Refactoring Changes")
    print("="*60)

    tests = []

    # Check that CYP3A4 specific features are removed
    try:
        from featurizer.molecule_featurizer.molecular_feature import MolecularFeatureExtractor
        extractor = MolecularFeatureExtractor()

        # Check that CYP3A4 attributes don't exist
        assert not hasattr(extractor, 'CYP3A4_INHIBITOR_SMARTS'), "CYP3A4_INHIBITOR_SMARTS still exists"
        assert not hasattr(extractor, 'CYP3A4_SUBSTRATE_SMARTS'), "CYP3A4_SUBSTRATE_SMARTS still exists"
        print("✓ CYP3A4 specific SMARTS removed")
        tests.append(True)

        # Check that new atom composition method exists
        assert hasattr(extractor, 'get_atom_composition_features'), "get_atom_composition_features missing"
        print("✓ New atom composition method exists")
        tests.append(True)

        # Check that extract_all_features accepts mol object
        import inspect
        sig = inspect.signature(extractor.extract_all_features)
        params = list(sig.parameters.keys())
        assert 'mol' in params, "extract_all_features doesn't accept 'mol' parameter"
        assert 'add_hs' in params, "extract_all_features missing 'add_hs' parameter"
        print("✓ extract_all_features properly refactored for mol objects")
        tests.append(True)

    except Exception as e:
        print(f"❌ Refactoring verification failed: {e}")
        tests.append(False)

    return all(tests)

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" Featurizer Package Refactoring Test")
    print("="*70)

    results = []

    # Run import tests
    results.append(("Import Tests", test_imports()))

    # Run molecular feature tests
    results.append(("Molecular Features", test_molecular_features()))

    # Run refactoring verification
    results.append(("Refactoring Changes", test_refactoring_changes()))

    # Summary
    print("\n" + "="*70)
    print(" Test Summary")
    print("="*70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! The refactoring is successful.")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
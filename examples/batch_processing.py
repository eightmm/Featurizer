#!/usr/bin/env python3
"""
Batch processing example for multiple PDB files.
"""

import os
from pathlib import Path
from protein_featurizer import Featurizer


def batch_processing_example():
    """
    Process multiple PDB files in batch.
    """
    print("Batch Processing Example")
    print("=" * 50)

    # Initialize featurizer
    featurizer = Featurizer(standardize=True, keep_hydrogens=False)

    # Example: Process all PDB files in a directory
    input_dir = "path/to/pdb/files"
    output_dir = "path/to/output/features"

    # For demonstration, we'll use a list of example files
    pdb_files = [
        "protein1.pdb",
        "protein2.pdb",
        "protein3.pdb",
    ]

    print("\nExample batch processing code:")
    print("-" * 40)
    print("""
    # Process multiple PDB files
    pdb_files = ['protein1.pdb', 'protein2.pdb', 'protein3.pdb']

    results = featurizer.extract_batch(
        pdb_files,
        output_dir='features/',
        skip_existing=True,
        verbose=True
    )

    # Check results
    for file_name, result in results.items():
        if result:
            print(f"✓ {file_name}: {result}")
        else:
            print(f"✗ {file_name}: Failed")
    """)


def directory_processing_example():
    """
    Process all PDB files in a directory.
    """
    print("\n" + "=" * 50)
    print("Directory Processing Example")
    print("=" * 50)

    print("\nExample code for processing entire directory:")
    print("-" * 40)
    print("""
    from pathlib import Path
    from protein_featurizer import Featurizer

    # Find all PDB files in directory
    input_dir = Path('data/pdbs')
    pdb_files = list(input_dir.glob('*.pdb'))

    # Process all files
    featurizer = Featurizer()
    results = featurizer.extract_batch(
        [str(f) for f in pdb_files],
        output_dir='data/features/',
        skip_existing=True,
        verbose=True
    )

    print(f"Processed {len(results)} files")
    """)


def parallel_processing_example():
    """
    Example of processing files with different settings.
    """
    print("\n" + "=" * 50)
    print("Advanced Batch Processing")
    print("=" * 50)

    print("\nProcessing with different settings:")
    print("-" * 40)
    print("""
    # Process different groups with different settings

    # Group 1: Standard proteins
    standard_featurizer = Featurizer(standardize=True)
    standard_results = standard_featurizer.extract_batch(
        standard_proteins,
        output_dir='features/standard/'
    )

    # Group 2: Already clean proteins
    clean_featurizer = Featurizer(standardize=False)
    clean_results = clean_featurizer.extract_batch(
        clean_proteins,
        output_dir='features/clean/'
    )

    # Group 3: Proteins with hydrogens
    hydrogen_featurizer = Featurizer(keep_hydrogens=True)
    hydrogen_results = hydrogen_featurizer.extract_batch(
        hydrogen_proteins,
        output_dir='features/with_hydrogens/'
    )
    """)


def error_handling_example():
    """
    Example of handling errors in batch processing.
    """
    print("\n" + "=" * 50)
    print("Error Handling in Batch Processing")
    print("=" * 50)

    print("\nRobust batch processing with error handling:")
    print("-" * 40)
    print("""
    from protein_featurizer import Featurizer

    featurizer = Featurizer()
    pdb_files = ['good.pdb', 'corrupt.pdb', 'missing.pdb']

    # Process with error handling
    results = featurizer.extract_batch(
        pdb_files,
        output_dir='features/',
        skip_existing=True,
        verbose=True
    )

    # Analyze results
    successful = [f for f, r in results.items() if r is not None]
    failed = [f for f, r in results.items() if r is None]

    print(f"Success: {len(successful)} files")
    print(f"Failed: {len(failed)} files")

    if failed:
        print("Failed files:", failed)
    """)


if __name__ == "__main__":
    batch_processing_example()
    directory_processing_example()
    parallel_processing_example()
    error_handling_example()

    print("\n" + "=" * 50)
    print("Note: Replace example paths with your actual file paths")
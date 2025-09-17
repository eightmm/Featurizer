#!/usr/bin/env python3
"""
Main script for protein feature extraction pipeline.

This script orchestrates the PDB standardization and feature extraction process,
providing a complete pipeline from raw PDB files to extracted features.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

from .pdb_standardizer import PDBStandardizer
from .residue_featurizer import ResidueFeaturizer


def process_pdb(input_pdb: str,
                output_path: Optional[str] = None,
                standardize: bool = True,
                keep_hydrogens: bool = False,
                verbose: bool = True) -> dict:
    """
    Process a PDB file and extract features.

    Args:
        input_pdb: Path to input PDB file
        output_path: Optional path to save features
        standardize: Whether to standardize the PDB first
        keep_hydrogens: Whether to keep hydrogen atoms during standardization
        verbose: Whether to print progress messages

    Returns:
        Dictionary containing extracted features
    """
    try:
        # Determine PDB file to process
        if standardize:
            if verbose:
                print(f"Standardizing PDB file: {input_pdb}")

            # Create temporary file for standardized PDB
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb_path = tmp_file.name

            # Standardize the PDB
            standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)
            pdb_to_process = standardizer.standardize(input_pdb, tmp_pdb_path)

            if verbose:
                print(f"PDB standardized successfully")
        else:
            pdb_to_process = input_pdb

        # Extract features
        if verbose:
            print(f"Extracting features from: {pdb_to_process}")

        featurizer = ResidueFeaturizer(pdb_to_process)
        node_features, edge_features = featurizer.get_features()

        # Combine features
        features = {
            'node': node_features,
            'edge': edge_features,
            'metadata': {
                'input_file': input_pdb,
                'standardized': standardize,
                'hydrogens_removed': not keep_hydrogens if standardize else None
            }
        }

        # Save features if output path specified
        if output_path:
            torch.save(features, output_path)
            if verbose:
                print(f"Features saved to: {output_path}")

        # Clean up temporary file
        if standardize:
            os.unlink(tmp_pdb_path)

        return features

    except Exception as e:
        if verbose:
            print(f"Error processing {input_pdb}: {str(e)}", file=sys.stderr)
        raise


def batch_process(input_dir: str, output_dir: str,
                  pattern: str = "*.pdb",
                  standardize: bool = True,
                  keep_hydrogens: bool = False,
                  skip_existing: bool = True,
                  verbose: bool = True):
    """
    Process multiple PDB files in batch.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save feature files
        pattern: Glob pattern for PDB files
        standardize: Whether to standardize PDBs first
        keep_hydrogens: Whether to keep hydrogen atoms
        skip_existing: Whether to skip already processed files
        verbose: Whether to print progress messages
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PDB files
    pdb_files = list(input_path.glob(pattern))

    if verbose:
        print(f"Found {len(pdb_files)} PDB files to process")

    processed = 0
    skipped = 0
    failed = 0

    for pdb_file in pdb_files:
        # Determine output file path
        output_file = output_path / f"{pdb_file.stem}_features.pt"

        # Skip if already exists
        if skip_existing and output_file.exists():
            if verbose:
                print(f"Skipping {pdb_file.name} (already processed)")
            skipped += 1
            continue

        try:
            # Process the PDB file
            process_pdb(
                str(pdb_file),
                str(output_file),
                standardize=standardize,
                keep_hydrogens=keep_hydrogens,
                verbose=False
            )
            processed += 1
            if verbose:
                print(f"Processed: {pdb_file.name} → {output_file.name}")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"Failed: {pdb_file.name} - {str(e)}", file=sys.stderr)

    # Print summary
    if verbose:
        print("\n" + "="*50)
        print(f"Batch processing complete:")
        print(f"  Processed: {processed}")
        print(f"  Skipped:   {skipped}")
        print(f"  Failed:    {failed}")
        print(f"  Total:     {len(pdb_files)}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Protein Feature Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDB file
  python main.py input.pdb -o features.pt

  # Process without standardization
  python main.py input.pdb -o features.pt --no-standardize

  # Keep hydrogen atoms
  python main.py input.pdb -o features.pt --keep-hydrogens

  # Batch process directory
  python main.py --batch input_dir/ output_dir/

  # Batch process with custom pattern
  python main.py --batch input_dir/ output_dir/ --pattern "**/*_protein.pdb"
        """
    )

    # Mode selection
    parser.add_argument('input', nargs='?', help='Input PDB file (for single file mode)')
    parser.add_argument('-o', '--output', help='Output file path for features')

    # Batch mode
    parser.add_argument('--batch', nargs=2, metavar=('INPUT_DIR', 'OUTPUT_DIR'),
                       help='Batch process PDB files from INPUT_DIR to OUTPUT_DIR')
    parser.add_argument('--pattern', default='*.pdb',
                       help='Glob pattern for batch mode (default: *.pdb)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Process files even if output already exists')

    # Processing options
    parser.add_argument('--no-standardize', action='store_true',
                       help='Skip PDB standardization step')
    parser.add_argument('--keep-hydrogens', action='store_true',
                       help='Keep hydrogen atoms (default: remove)')

    # Output options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.batch:
        parser.error("Either provide an input PDB file or use --batch mode")

    if args.input and args.batch:
        parser.error("Cannot use both single file and batch mode")

    verbose = not args.quiet

    # Execute appropriate mode
    if args.batch:
        input_dir, output_dir = args.batch
        batch_process(
            input_dir,
            output_dir,
            pattern=args.pattern,
            standardize=not args.no_standardize,
            keep_hydrogens=args.keep_hydrogens,
            skip_existing=not args.no_skip,
            verbose=verbose
        )
    else:
        # Single file mode
        if not args.output:
            # Default output name
            input_path = Path(args.input)
            args.output = f"{input_path.stem}_features.pt"

        try:
            features = process_pdb(
                args.input,
                args.output,
                standardize=not args.no_standardize,
                keep_hydrogens=args.keep_hydrogens,
                verbose=verbose
            )

            if verbose:
                print(f"\nFeature extraction complete!")
                print(f"Number of residues: {len(features['node']['coord'])}")
                print(f"Number of edges: {len(features['edge']['edges'][0])}")

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
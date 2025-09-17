"""
Protein Featurizer Module

A comprehensive toolkit for extracting structural features from protein PDB files.
"""

from .pdb_standardizer import PDBStandardizer, standardize_pdb
from .residue_featurizer import ResidueFeaturizer
from .main import process_pdb, batch_process

__version__ = "0.2.0"
__author__ = "Jaemin Sim"

__all__ = [
    "PDBStandardizer",
    "ResidueFeaturizer",
    "standardize_pdb",
    "process_pdb",
    "batch_process",
    "Featurizer",  # Main API class
]


class Featurizer:
    """
    High-level API for protein feature extraction.

    This class provides a simple, unified interface for the complete
    feature extraction pipeline.

    Examples:
        >>> # Basic usage
        >>> from protein_featurizer import Featurizer
        >>> featurizer = Featurizer()
        >>> features = featurizer.extract("protein.pdb")

        >>> # Without standardization
        >>> featurizer = Featurizer(standardize=False)
        >>> features = featurizer.extract("clean.pdb")

        >>> # Custom options
        >>> featurizer = Featurizer(keep_hydrogens=True)
        >>> features = featurizer.extract("protein.pdb", save_to="features.pt")
    """

    def __init__(self, standardize: bool = True, keep_hydrogens: bool = False):
        """
        Initialize the Featurizer.

        Args:
            standardize: Whether to standardize PDB files before feature extraction
            keep_hydrogens: Whether to keep hydrogen atoms during standardization
        """
        self.standardize = standardize
        self.keep_hydrogens = keep_hydrogens
        self._standardizer = None
        self._featurizer = None

        if self.standardize:
            self._standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)

    def extract(self, pdb_file: str, save_to: str = None) -> dict:
        """
        Extract features from a PDB file.

        Args:
            pdb_file: Path to the PDB file
            save_to: Optional path to save the extracted features

        Returns:
            Dictionary containing node and edge features

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If feature extraction fails
        """
        import os
        import tempfile
        import torch

        # Check if file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        try:
            # Standardize if requested
            if self.standardize:
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                    tmp_pdb = tmp_file.name

                self._standardizer.standardize(pdb_file, tmp_pdb)
                pdb_to_process = tmp_pdb
            else:
                pdb_to_process = pdb_file

            # Extract features
            self._featurizer = ResidueFeaturizer(pdb_to_process)
            node_features, edge_features = self._featurizer.get_features()

            # Package features
            features = {
                'node': node_features,
                'edge': edge_features,
                'metadata': {
                    'input_file': pdb_file,
                    'standardized': self.standardize,
                    'hydrogens_removed': not self.keep_hydrogens if self.standardize else None
                }
            }

            # Save if requested
            if save_to:
                torch.save(features, save_to)

            # Cleanup
            if self.standardize:
                os.unlink(pdb_to_process)

            return features

        except Exception as e:
            raise ValueError(f"Failed to extract features from {pdb_file}: {str(e)}")

    def extract_batch(self, pdb_files: list, output_dir: str = None,
                     skip_existing: bool = True, verbose: bool = True) -> dict:
        """
        Extract features from multiple PDB files.

        Args:
            pdb_files: List of PDB file paths
            output_dir: Directory to save feature files (optional)
            skip_existing: Whether to skip files that already have features
            verbose: Whether to print progress

        Returns:
            Dictionary mapping file names to features or output paths
        """
        import os
        from pathlib import Path

        results = {}

        for i, pdb_file in enumerate(pdb_files):
            file_name = Path(pdb_file).stem

            # Determine output path if saving
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{file_name}_features.pt")

                # Skip if exists
                if skip_existing and os.path.exists(output_path):
                    if verbose:
                        print(f"[{i+1}/{len(pdb_files)}] Skipping {file_name} (already exists)")
                    results[file_name] = output_path
                    continue

            try:
                if verbose:
                    print(f"[{i+1}/{len(pdb_files)}] Processing {file_name}...")

                features = self.extract(pdb_file, save_to=output_path)
                results[file_name] = output_path if output_path else features

            except Exception as e:
                if verbose:
                    print(f"[{i+1}/{len(pdb_files)}] Failed {file_name}: {str(e)}")
                results[file_name] = None

        return results

    @classmethod
    def from_clean_pdb(cls, pdb_file: str) -> dict:
        """
        Extract features from an already clean/standardized PDB file.

        Args:
            pdb_file: Path to clean PDB file

        Returns:
            Dictionary containing features
        """
        featurizer = cls(standardize=False)
        return featurizer.extract(pdb_file)

    @staticmethod
    def standardize_only(input_pdb: str, output_pdb: str,
                        keep_hydrogens: bool = False) -> str:
        """
        Only standardize a PDB file without feature extraction.

        Args:
            input_pdb: Input PDB file path
            output_pdb: Output PDB file path
            keep_hydrogens: Whether to keep hydrogen atoms

        Returns:
            Path to standardized PDB file
        """
        return standardize_pdb(input_pdb, output_pdb,
                              remove_hydrogens=not keep_hydrogens)
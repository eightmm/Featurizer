"""
Setup script for Protein Featurizer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="protein-featurizer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for extracting structural features from protein PDB files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eightmm/protein-featurizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "protein-featurizer=protein_featurizer.main:main",
            "pdb-standardize=protein_featurizer.pdb_standardizer:main",
            "extract-features=protein_featurizer.residue_featurizer:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="protein pdb feature-extraction bioinformatics structural-biology machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/eightmm/protein-featurizer/issues",
        "Source": "https://github.com/eightmm/protein-featurizer",
    },
)
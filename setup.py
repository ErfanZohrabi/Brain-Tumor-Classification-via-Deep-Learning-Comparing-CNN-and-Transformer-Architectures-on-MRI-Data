#!/usr/bin/env python3
"""
Setup script for Brain Tumor Classifier package.
"""

from setuptools import setup, find_packages

setup(
    name="brain_tumor_classifier",
    version="1.0.0",
    description="Brain Tumor Classification using Deep Learning",
    author="Erfan Zohrabi",
    author_email="Erfan.zohrabi@studio.unibo.it",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.1.0",
        "tqdm>=4.50.0",
        "pillow>=8.0.0",
        "timm>=0.9.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "brain-tumor-classifier=brain_tumor_classifier_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 
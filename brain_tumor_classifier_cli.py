#!/usr/bin/env python3
"""
Command-line interface for Brain Tumor Classification.
"""

import sys
import argparse
from brain_tumor_classifier.main import main as run_training

def main():
    """Entry point for the command-line interface."""
    # Just redirect to the main function
    sys.exit(run_training())

if __name__ == "__main__":
    main() 
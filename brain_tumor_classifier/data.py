"""
Data handling utilities for Brain Tumor Classification.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image
from sklearn.model_selection import StratifiedKFold, train_test_split

def mat_to_sample(mat_path: Path) -> Tuple[PIL.Image.Image, int]:
    """
    Loads an image and label from a .mat file.
    
    Args:
        mat_path: Path to the .mat file
        
    Returns:
        Tuple containing the image as a PIL Image and the label (0-based)
    """
    with h5py.File(mat_path, "r") as f:
        # Data structure: cjdata/image, cjdata/label, cjdata/Readme
        # Label is 1-based, convert to 0-based
        img = np.array(f["cjdata/image"], dtype=np.float32)
        label = int(f["cjdata/label"][()].item()) - 1
    # Normalize image to 0-255 range
    img = 255 * (img - img.min()) / (np.ptp(img) + 1e-5)
    return PIL.Image.fromarray(img.astype(np.uint8)), label


class MRIDataset(Dataset):
    """
    Custom Dataset for MRI images.
    
    Args:
        samples: List of tuples containing (image, label)
        train: Whether this is a training dataset (enables data augmentation)
    """
    def __init__(self, samples: List[Tuple[PIL.Image.Image, int]], train: bool):
        # Transformations: Data augmentation for training, only resizing and normalization for validation/testing
        aug = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ]
        base = [
            # ViT models often use 384x384 input size for optimal performance with SWAG weights
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            # Grayscale image needs to be repeated across 3 channels for pretrained models
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
            # Normalize with standard values
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.tf = transforms.Compose((aug if train else []) + base)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, lbl = self.samples[i]
        return self.tf(img), torch.tensor(lbl, dtype=torch.long)


def load_dataset(
    data_path: Path, 
    limit: int = 0, 
    verbose: bool = True
) -> Tuple[List[Tuple[PIL.Image.Image, int]], int]:
    """
    Extracts and loads samples from dataset ZIP files.
    
    Args:
        data_path: Path to the directory containing dataset ZIP files
        limit: Maximum number of .mat files to extract per ZIP (0 = no limit)
        verbose: Whether to print progress information
    
    Returns:
        Tuple containing list of samples and number of classes
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {data_path}")

    zips = sorted(data_path.glob("brainTumorDataPublic_*.zip"))
    if not zips:
        print(f"Warning: No ZIP files found matching 'brainTumorDataPublic_*.zip' in {data_path}")
        # Proceed with empty samples, will raise error later if no samples
    elif verbose:
        print(f"Found {len(zips)} ZIP files.")

    tmp = Path("._tmp_extract")  # Temporary extraction directory
    # Clean up previous temporary directory if it exists
    if tmp.exists():
        if verbose:
            print(f"Cleaning up temporary directory: {tmp}")
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=False)  # Create a new temporary directory

    samples: List[Tuple[PIL.Image.Image, int]] = []

    for z in zips:
        try:
            with zipfile.ZipFile(z, 'r') as arc:
                # List .mat files within the zip, skipping directory entries
                mat_files = [m for m in arc.namelist() if m.endswith(".mat") and not m.endswith('/')]
                if limit > 0:
                    mat_files = mat_files[:limit]

                if verbose:
                    print(f"Processing {z.name}: {len(mat_files)} files")

                for m in mat_files:
                    try:
                        # Extract the specific .mat file
                        arc.extract(m, tmp)
                        # Process the extracted .mat file
                        sample = mat_to_sample(tmp / m)
                        samples.append(sample)
                    except Exception as e:
                        print(f"Error processing file {m} in {z.name}: {e}")
                    finally:
                        # Clean up the extracted file immediately
                        extracted_file = tmp / m
                        if extracted_file.exists():
                            try:
                                os.remove(extracted_file)
                            except OSError as e:
                                print(f"Error removing extracted file {extracted_file}: {e}")
        except zipfile.BadZipFile:
            print(f"Error: File {z.name} is not a valid zip file.")
        except Exception as e:
            print(f"An error occurred while processing {z.name}: {e}")

    # Clean up the temporary directory after processing all zips
    if tmp.exists():
        if verbose:
            print(f"Final cleanup of temporary directory: {tmp}")
        try:
            shutil.rmtree(tmp)
        except OSError as e:
            print(f"Error during final cleanup of {tmp}: {e}")

    if not samples:
        raise RuntimeError("No samples found. Please check the dataset path and ZIP files.")

    # Auto-detect the true number of classes
    num_classes = len(set(lbl for _, lbl in samples))
    if verbose:
        print(f"Loaded {len(samples)} samples across {num_classes} classes.")

    return samples, num_classes


def create_data_loaders(
    samples: List[Tuple[PIL.Image.Image, int]],
    batch_size: int = 16,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 2
) -> Dict[str, Any]:
    """
    Creates train/validation/test data loaders and datasets.
    
    Args:
        samples: List of samples (image, label)
        batch_size: Batch size for the data loaders
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
        num_workers: Number of worker threads for data loading
        
    Returns:
        Dictionary containing data loaders and related information
    """
    # Extract labels
    y_all = np.array([lbl for _, lbl in samples])
    
    # Split data into training+validation and test sets
    train_val_idx, test_idx = train_test_split(
        np.arange(len(samples)),
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed
    )
    
    # Further split training+validation into training and validation
    y_train_val = np.array([y_all[i] for i in train_val_idx])
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=val_size,
        stratify=y_train_val,
        random_state=random_seed
    )
    
    # Convert indices to actual samples
    train_samples = [samples[train_val_idx[i]] for i in train_idx]
    val_samples = [samples[train_val_idx[i]] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    # Create datasets
    train_dataset = MRIDataset(train_samples, train=True)
    val_dataset = MRIDataset(val_samples, train=False)
    test_dataset = MRIDataset(test_samples, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }


def create_kfold_datasets(
    samples: List[Tuple[PIL.Image.Image, int]],
    n_folds: int = 5,
    batch_size: int = 16,
    num_workers: int = 2,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Creates datasets and loaders for k-fold cross-validation.
    
    Args:
        samples: List of samples (image, label)
        n_folds: Number of folds for cross-validation
        batch_size: Batch size for the data loaders
        num_workers: Number of worker threads for data loading
        random_seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries, each containing train/val loaders for a fold
    """
    y = np.array([lbl for _, lbl in samples])
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=random_seed)
    
    fold_data = []
    
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        # Create datasets for this fold
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        train_dataset = MRIDataset(train_samples, train=True)
        val_dataset = MRIDataset(val_samples, train=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        fold_data.append({
            "train_loader": train_loader,
            "val_loader": val_loader,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset
        })
    
    return fold_data 
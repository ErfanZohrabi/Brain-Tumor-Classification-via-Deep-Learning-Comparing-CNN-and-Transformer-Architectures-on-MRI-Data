"""
Main script for training and evaluating brain tumor classification models.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

from brain_tumor_classifier.data import load_dataset, create_data_loaders, create_kfold_datasets
from brain_tumor_classifier.models import create_model
from brain_tumor_classifier.train import train_model, cross_validate
from brain_tumor_classifier.metrics import SciMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the brain tumor dataset ZIP files')
    parser.add_argument('--quick_limit', type=int, default=0,
                        help='Limit the number of .mat files per ZIP (0 = no limit)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='se_resnet50',
                        help='Model architecture to use. Options:\n'
                             '1. Primary comparison models: se_resnet50, swin_hybrid\n'
                             '2. Benchmark models: convnext_base, efficientnet_b0, efficientnet_b3, densenet121,\n'
                             '   regnety_032, maxvit_tiny_rw_224, inception_v3, coat_lite_medium\n'
                             '3. Others: torchvision models (vit_torchvision, etc.) or any other timm model name')
    parser.add_argument('--img_size', type=int, default=384,
                        help='Input image size, particularly for Vision Transformers and Swin Transformers. '
                             'Default is 384 (from data loader), but some Swin models (like Swin-Base-224) expect 224 for pretraining.')
    parser.add_argument('--swin_model_name', type=str, default='swin_base_patch4_window7_224',
                        help='Specify the Swin Transformer backbone name from timm when using --model swin_hybrid.')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Number of epochs to wait before early stopping (0 = no early stopping)')
    
    # Cross-validation parameters
    parser.add_argument('--cv', action='store_true',
                        help='Enable cross-validation')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu, or None for auto-detection)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to store output files')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of experiment (default: auto-generated based on model and date)')
    
    # Class names (optional)
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['meningioma', 'glioma', 'pituitary'],
                        help='Names of classes (adjust as needed)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Extra deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name=None):
    """
    Get PyTorch device to use.
    
    Args:
        device_name: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
    
    Returns:
        PyTorch device
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_name == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif device_name == 'cpu':
        return torch.device('cpu')
    
    # Auto-detection
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_optimizer(model, args):
    """Create optimizer for training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


def create_scheduler(optimizer, args, n_batches):
    """Create learning rate scheduler."""
    # One cycle learning rate scheduler
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=n_batches,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device to use
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = Path(args.data_dir)
    samples, num_classes = load_dataset(data_path, args.quick_limit, args.verbose)
    
    # Adjust class names if needed
    if len(args.class_names) != num_classes:
        print(f"Warning: Provided {len(args.class_names)} class names but detected {num_classes} classes.")
        print(f"Using default class names: Class 0, Class 1, etc.")
        args.class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Create scientific metrics object
    sci_metrics = SciMetrics(class_names=args.class_names)
    
    # Handle cross-validation case
    if args.cv:
        # Create datasets for k-fold cross-validation
        fold_data = create_kfold_datasets(
            samples, 
            n_folds=args.folds,
            batch_size=args.batch_size,
            num_workers=2,
            random_seed=args.seed
        )
        
        # Define model creation function
        def model_fn():
            return create_model(args.model, num_classes, pretrained=True, 
                                img_size=args.img_size, swin_model_name=args.swin_model_name)
        
        # Define optimizer creation function
        def optimizer_fn(model):
            return create_optimizer(model, args)
        
        # Define scheduler creation function
        def scheduler_fn(optimizer):
            # Use a basic scheduler for cross-validation
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs
            )
        
        # Perform cross-validation
        cv_results = cross_validate(
            model_fn=model_fn,
            fold_data=fold_data,
            criterion=nn.CrossEntropyLoss(),
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            device=device,
            num_epochs=args.epochs,
            early_stopping=args.early_stopping,
            mixed_precision=args.mixed_precision,
            verbose=args.verbose,
            checkpoint_path=os.path.join(output_dir, 'model')
        )
        
        # Get best model from cross-validation
        best_model = cv_results['best_fold_result']['model']
        best_val_score = cv_results['best_fold_result']['best_val_score']
        
        print(f"\nCross-validation complete. Best validation score: {best_val_score:.4f}")
        print(f"Average validation score: {cv_results['avg_val_score']:.4f} ± {cv_results['std_val_score']:.4f}")
        
        # Save cross-validation results
        torch.save(cv_results, os.path.join(output_dir, 'cv_results.pt'))
        
    else:
        # Create regular train/val/test data loaders
        data = create_data_loaders(
            samples,
            batch_size=args.batch_size,
            test_size=0.2,
            val_size=0.2,
            random_seed=args.seed
        )
        
        # Create model
        model = create_model(args.model, num_classes, pretrained=True, 
                            img_size=args.img_size, swin_model_name=args.swin_model_name)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, args)
        scheduler = create_scheduler(optimizer, args, len(data['train_loader']))
        
        # Train model
        train_result = train_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            early_stopping=args.early_stopping,
            mixed_precision=args.mixed_precision,
            verbose=args.verbose,
            checkpoint_path=os.path.join(output_dir, 'model'),
            class_names=args.class_names
        )
        
        # Get best model from training
        best_model = train_result['model']
        
        # Plot training history
        sci_metrics.plot_training_history(
            train_result['history'],
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    
    if not args.cv:
        # For regular training, we already have test data
        test_loader = data['test_loader']
    else:
        # For cross-validation, create a test dataset from scratch
        _, test_samples = train_test_split(
            samples,
            test_size=0.2,
            random_state=args.seed,
            stratify=[lbl for _, lbl in samples]
        )
        
        from torch.utils.data import DataLoader
        from brain_tumor_classifier.data import MRIDataset
        
        test_dataset = MRIDataset(test_samples, train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    # Collect predictions on test set
    best_model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Aggregate results
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # Generate performance report
    sci_metrics.create_performance_report(
        all_targets,
        all_preds,
        all_probs,
        output_dir=output_dir,
        prefix='test_'
    )
    
    # Save test results
    torch.save({
        'targets': all_targets,
        'predictions': all_preds,
        'probabilities': all_probs
    }, os.path.join(output_dir, 'test_results.pt'))
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 
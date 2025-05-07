"""
Training utilities for Brain Tumor Classification.
"""

from typing import Dict, List, Tuple, Optional, Union, Callable
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from .metrics import auc_macro


def step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    train: bool,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    autocast_cfg: Optional[Dict] = None
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Perform a single training or evaluation step.
    
    Args:
        model: The model to use
        batch: A tuple of (inputs, targets)
        criterion: Loss function
        train: Whether this is a training step
        device: Device to use for computation
        scaler: Optional GradScaler for mixed precision training
        autocast_cfg: Configuration for autocast
        
    Returns:
        Tuple of (loss, logits, targets)
    """
    # Move data to device
    inputs, targets = [x.to(device) for x in batch]
    
    # Context manager for enabling/disabling gradient calculation
    ctx = torch.enable_grad() if train else torch.no_grad()
    
    # Autocast for mixed precision
    if autocast_cfg is None:
        autocast_cfg = {}
    
    # Forward pass
    with ctx, torch.autocast(**autocast_cfg):
        logits = model(inputs)
        loss = criterion(logits, targets)
    
    return loss, logits, targets


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    autocast_cfg: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Run a single epoch of training or evaluation.
    
    Args:
        model: The model to train/evaluate
        data_loader: DataLoader providing the data
        criterion: Loss function
        device: Device to use for computation
        optimizer: Optional optimizer (if None, evaluation mode is used)
        scheduler: Optional learning rate scheduler
        scaler: Optional GradScaler for mixed precision training
        autocast_cfg: Configuration for autocast
        verbose: Whether to display a progress bar
        
    Returns:
        Dictionary with epoch statistics
    """
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    # Setup progress bar if verbose
    data_iter = tqdm(data_loader) if verbose else data_loader
    
    # Iterate through batches
    for batch in data_iter:
        # Step (forward pass and optionally backward pass)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        
        loss, logits, targets = step(
            model, batch, criterion, is_train, device, scaler, autocast_cfg
        )
        
        # Backward pass and optimization step
        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            # Step LR scheduler if it's an epoch-based scheduler
            if scheduler is not None and hasattr(scheduler, 'step_after_batch') and scheduler.step_after_batch:
                scheduler.step()
                
        # Collect statistics
        total_loss += loss.item() * targets.size(0)
        
        # Convert outputs to predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store targets, predictions, and probabilities for metrics calculation
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
    # Step LR scheduler if it's an epoch-based scheduler
    if is_train and scheduler is not None and not hasattr(scheduler, 'step_after_batch'):
        scheduler.step()
        
    # Compute epoch statistics
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    accuracy = (all_targets == all_preds).mean()
    num_samples = len(all_targets)
    avg_loss = total_loss / num_samples
    
    # Calculate AUC if there are enough samples
    try:
        auc = auc_macro(all_targets, all_probs)
    except Exception as e:
        print(f"Could not compute AUC: {e}")
        auc = np.nan
        
    # Return statistics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'targets': all_targets,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_epochs: int = 10,
    early_stopping: int = 0,
    mixed_precision: bool = False,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    checkpoint_freq: int = 1,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Train a model with validation.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Optional learning rate scheduler
        device: Device to use for training
        num_epochs: Maximum number of epochs to train for
        early_stopping: Number of epochs to wait for improvement before stopping (0 = no early stopping)
        mixed_precision: Whether to use mixed precision training
        verbose: Whether to display progress
        checkpoint_path: Path to save model checkpoints (or None to skip saving)
        checkpoint_freq: How often to save checkpoints (in epochs)
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary with training history and best model state
    """
    # Move model to device
    model = model.to(device)
    
    # Configure mixed precision if requested
    scaler = None
    autocast_cfg = {'enabled': False}
    
    if mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        autocast_cfg = {'enabled': True, 'dtype': torch.float16, 'device_type': 'cuda'}
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': []
    }
    
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    if verbose:
        print("Starting training...")
        
    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Training phase
        if verbose:
            print(f"\nEpoch {epoch}/{num_epochs}")
            
        train_stats = run_epoch(
            model, train_loader, criterion, device, 
            optimizer, scheduler, scaler, autocast_cfg, verbose
        )
        
        # Validation phase
        val_stats = run_epoch(
            model, val_loader, criterion, device,
            None, None, scaler, autocast_cfg, verbose
        )
        
        # Record statistics
        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['accuracy'])
        history['train_auc'].append(train_stats['auc'])
        history['val_loss'].append(val_stats['loss'])
        history['val_acc'].append(val_stats['accuracy'])
        history['val_auc'].append(val_stats['auc'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print metrics
        if verbose:
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_stats['loss']:.4f}, Accuracy: {train_stats['accuracy']:.4f}, AUC: {train_stats['auc']:.4f}")
            print(f"  Val Loss: {val_stats['loss']:.4f}, Accuracy: {val_stats['accuracy']:.4f}, AUC: {val_stats['auc']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpoint
        if checkpoint_path and epoch % checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_stats': train_stats,
                'val_stats': val_stats,
                'history': history
            }
            torch.save(checkpoint, f"{checkpoint_path}_epoch_{epoch}.pt")
            
        # Check if this is the best model so far
        # Use AUC as primary metric, falling back to accuracy if AUC is NaN
        val_score = np.nan_to_num(val_stats['auc'], nan=val_stats['accuracy'])
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict().copy()
            
            if verbose:
                print(f"  New best model! Score: {val_score:.4f}")
                
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if early_stopping > 0 and patience_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break
            
    # Final checkpoint
    if checkpoint_path:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'best_val_score': best_val_score
        }, f"{checkpoint_path}_final.pt")
        
        # Also save the best model separately
        torch.save(best_model_state, f"{checkpoint_path}_best.pt")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return {
        'model': model,
        'history': history,
        'best_val_score': best_val_score,
        'best_model_state': best_model_state
    }


def cross_validate(
    model_fn: Callable,
    fold_data: List[Dict],
    criterion: nn.Module,
    optimizer_fn: Callable,
    scheduler_fn: Optional[Callable] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_epochs: int = 10,
    early_stopping: int = 0,
    mixed_precision: bool = False,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None
) -> Dict:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_fn: Function to create a model
        fold_data: List of dictionaries containing train/val loaders for each fold
        criterion: Loss function
        optimizer_fn: Function to create an optimizer given a model
        scheduler_fn: Optional function to create a learning rate scheduler
        device: Device to use for training
        num_epochs: Maximum number of epochs to train for
        early_stopping: Number of epochs to wait for improvement before stopping (0 = no early stopping)
        mixed_precision: Whether to use mixed precision training
        verbose: Whether to display progress
        checkpoint_path: Path to save model checkpoints (or None to skip saving)
        
    Returns:
        Dictionary with cross-validation results
    """
    num_folds = len(fold_data)
    cv_results = []
    
    for fold_idx, fold in enumerate(fold_data):
        if verbose:
            print(f"\n======= Fold {fold_idx + 1}/{num_folds} =======")
            
        # Create a new model for this fold
        model = model_fn()
        
        # Create optimizer and scheduler
        optimizer = optimizer_fn(model)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None
        
        # Setup checkpoint path for this fold
        fold_checkpoint_path = f"{checkpoint_path}_fold_{fold_idx + 1}" if checkpoint_path else None
        
        # Train the model on this fold
        fold_result = train_model(
            model=model,
            train_loader=fold['train_loader'],
            val_loader=fold['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            mixed_precision=mixed_precision,
            verbose=verbose,
            checkpoint_path=fold_checkpoint_path
        )
        
        cv_results.append(fold_result)
        
    # Compute average metrics across folds
    val_scores = [result['best_val_score'] for result in cv_results]
    avg_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    
    if verbose:
        print("\n======= Cross-Validation Results =======")
        print(f"Validation scores: {val_scores}")
        print(f"Average: {avg_val_score:.4f} ± {std_val_score:.4f}")
        
    # Find the best fold
    best_fold_idx = np.argmax(val_scores)
    best_fold_result = cv_results[best_fold_idx]
    
    return {
        'cv_results': cv_results,
        'avg_val_score': avg_val_score,
        'std_val_score': std_val_score,
        'best_fold_idx': best_fold_idx,
        'best_fold_result': best_fold_result
    } 
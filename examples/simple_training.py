#!/usr/bin/env python3
"""
Example script demonstrating how to use the Brain Tumor Classifier package.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

from brain_tumor_classifier.data import load_dataset, create_data_loaders
from brain_tumor_classifier.models import create_model
from brain_tumor_classifier.train import train_model
from brain_tumor_classifier.metrics import SciMetrics

# Configuration
DATA_DIR = "/path/to/your/dataset"  # Change this to your dataset path
OUTPUT_DIR = "example_output"
MODEL_TYPE = "vit"  # Options: 'vit', 'resnet', 'efficientnet'
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]  # Adjust as needed

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print(f"Loading dataset from {DATA_DIR}...")
samples, num_classes = load_dataset(Path(DATA_DIR), verbose=True)

# Create data loaders
print("Creating data loaders...")
data = create_data_loaders(
    samples,
    batch_size=BATCH_SIZE,
    test_size=0.2,
    val_size=0.2,
    random_seed=42
)

# Create model
print(f"Creating {MODEL_TYPE} model for {num_classes} classes...")
model = create_model(MODEL_TYPE, num_classes)

# Create optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Create scheduler (cosine annealing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=NUM_EPOCHS
)

# Initialize metrics tracker
metrics = SciMetrics(class_names=CLASS_NAMES[:num_classes])

# Train model
print(f"Training on {DEVICE}...")
train_result = train_model(
    model=model,
    train_loader=data['train_loader'],
    val_loader=data['val_loader'],
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    early_stopping=5,
    verbose=True,
    checkpoint_path=os.path.join(OUTPUT_DIR, "model"),
    class_names=CLASS_NAMES[:num_classes]
)

# Visualize training history
print("Plotting training history...")
metrics.plot_training_history(
    train_result['history'],
    save_path=os.path.join(OUTPUT_DIR, "training_history.png")
)

# Evaluate on test set
print("Evaluating on test set...")
model.eval()
all_targets = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, targets in data['test_loader']:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

# Convert to numpy arrays
import numpy as np
all_targets = np.concatenate(all_targets)
all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)

# Generate performance report
print("Generating performance report...")
test_metrics = metrics.create_performance_report(
    all_targets,
    all_preds,
    all_probs,
    output_dir=OUTPUT_DIR,
    prefix="test_"
)

print(f"Example completed. Results saved to: {OUTPUT_DIR}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1 Score (macro): {test_metrics['f1_macro']:.4f}")
if 'auc_macro' in test_metrics:
    print(f"Test AUC (macro): {test_metrics['auc_macro']:.4f}") 
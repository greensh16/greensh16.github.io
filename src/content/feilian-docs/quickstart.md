# Quickstart Guide

## Single GPU Training

The simplest way to start training:

```bash
python train.py
```

This uses default settings from `config.py`:
- 100 epochs
- Batch size 1
- Base channels 32 (~45M parameters)
- Mixed precision enabled
- Gradient loss enabled
- Early stopping (patience=20)

## Custom Training Run

### Smaller Model (Faster, Less Memory)

```bash
python train.py \
    --epochs 50 \
    --batch-size 1 \
    --base-channels 16 \
    --lr 0.001 \
    --use-amp \
    --use-scheduler \
    --early-stopping \
    --patience 15 \
    --checkpoint-dir checkpoints_small \
    --log-dir logs_small
```

### Larger Model (Better Accuracy, More Memory)

```bash
python train.py \
    --epochs 100 \
    --batch-size 1 \
    --base-channels 32 \
    --lr 0.0005 \
    --use-amp \
    --use-gradient-loss \
    --gradient-weight 0.1 \
    --use-scheduler \
    --early-stopping \
    --patience 20
```

## Resume Training

To resume from a checkpoint:

```bash
python train.py \
    --resume checkpoints/best_model.pt \
    --epochs 150  # Continue to epoch 150
```

The script automatically loads:
- Model weights
- Optimizer state
- Scheduler state
- Training history

## Multi-GPU Training (Local)

On a multi-GPU workstation:

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py \
    --epochs 50 \
    --batch-size 1 \
    --base-channels 32

# Or using python -m torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## Monitor Training

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs --port 6006

# Open in browser
# http://localhost:6006
```

View:
- Training/validation loss curves
- Learning rate schedule
- Gradient norms
- Custom metrics (MAE, RMSE, etc.)

### Watch Progress

```bash
# Follow training log
tail -f logs/train-*.out

# Check checkpoints
ls -lht checkpoints/
```

## Quick Training Examples

### Fast Experiment (5 minutes)

Test code changes quickly:

```bash
python train.py \
    --epochs 3 \
    --batch-size 1 \
    --base-channels 8 \
    --num-workers 2 \
    --val-split 0.1  # Use only 10% for validation (more training data)
```

### Overnight Run (8 hours)

```bash
python train.py \
    --epochs 100 \
    --batch-size 1 \
    --base-channels 32 \
    --use-amp \
    --use-gradient-loss \
    --early-stopping
```

### Hyperparameter Search

Try different learning rates:

```bash
for lr in 0.0001 0.0005 0.001 0.005; do
    python train.py \
        --epochs 30 \
        --lr $lr \
        --base-channels 16 \
        --checkpoint-dir checkpoints_lr${lr} \
        --log-dir logs_lr${lr}
done
```

## Common Options

### Model Architecture

```bash
--base-channels 16     # Small model (~11M params)
--base-channels 32     # Medium model (~45M params, default)
--base-channels 64     # Large model (~180M params)
--dropout 0.1          # Dropout rate
```

### Training

```bash
--epochs 100           # Number of epochs
--batch-size 1         # Batch size per GPU
--lr 0.001             # Learning rate
--weight-decay 1e-4    # AdamW weight decay
--clip-grad 1.0        # Gradient clipping threshold
```

### Loss Functions

```bash
--use-gradient-loss       # Enable physics-informed gradient loss
--gradient-weight 0.1     # Weight for gradient loss term
```

### Optimization

```bash
--use-amp                 # Enable mixed precision (faster, less memory)
--use-scheduler           # Enable OneCycleLR scheduler
```

### Data

```bash
--val-split 0.2           # Validation set fraction
--num-workers 4           # DataLoader workers
--normalize               # Normalize inputs
--norm-mode per_sample    # Normalization mode
--clip-range -1.0 10.0    # Clip wind speed values
```

### Early Stopping

```bash
--early-stopping          # Enable early stopping
--patience 20             # Epochs without improvement
```

### Output

```bash
--checkpoint-dir checkpoints  # Checkpoint directory
--log-dir logs                # Log directory
--use-tensorboard             # Enable TensorBoard logging
```

## Check Results

### View Training Progress

```bash
# Latest checkpoint
ls -t checkpoints/*.pt | head -1

# Best model metrics
python << EOF
import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
print(f"Best epoch: {ckpt['epoch']}")
print(f"Val loss: {ckpt['val_loss']:.4f}")
print(f"Val metrics: {ckpt['val_metrics']}")
EOF
```

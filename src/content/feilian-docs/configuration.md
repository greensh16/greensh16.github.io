## 9. Configuration System

### 9.1 config.py Structure

```python
# Data paths
DATA_ROOT = Path("data")
WIND_SPEED_DIR = DATA_ROOT / "wind_speed_filled"
MASK_DIR = DATA_ROOT / "mask_buildings"

# Model config
MODEL_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 32,
    "dropout": 0.1,
    "use_batchnorm": True,  # Ignored, using GroupNorm
}

# Training config
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "use_gradient_loss": True,
    "gradient_loss_weight": 0.1,
    ...
}

# Phase-based optimizations
PHASE3_CONFIG = {
    "use_charbonnier_loss": True,
    "improved_gradient_loss": True,
    "use_augmentation": True,
    ...
}
```

### 9.2 CLI Overrides

```bash
python train.py \
    --epochs 50 \              # Override TRAINING_CONFIG["epochs"]
    --batch-size 2 \
    --base-channels 16 \
    --lr 5e-4 \
    --use-scheduler \          # Boolean flag
    --checkpoint-dir my_ckpts
```

### 9.3 Hyperparameter Ranges

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `epochs` | 100 | 30-200 | Early stopping usually kicks in |
| `batch_size` | 1 | 1-4 | Larger → OOM, gradient accumulation as alternative |
| `base_channels` | 32 | 16-64 | 16=fast/small, 32=balanced, 64=slow/large |
| `lr` | 1e-3 | 1e-4 to 1e-2 | With OneCycleLR, max_lr |
| `weight_decay` | 1e-4 | 1e-5 to 1e-3 | AdamW decoupled weight decay |
| `gradient_weight` | 0.1 | 0-1 | λ for gradient loss |
| `clip_grad` | 1.0 | 0-5 | 0=disabled, 1=safe, 5=permissive |
| `patience` | 20 | 10-50 | Early stopping patience |

---


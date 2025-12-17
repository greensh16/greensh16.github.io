## 3. Data Pipeline

### 3.1 Dataset Structure

**File Organization**:
```
data/
├── wind_speed_filled/
│   ├── 15VF20_ws_filled.nc      # Wind angle 15°, variant VF20
│   ├── 30VF20_ws_filled.nc      # Wind angle 30°
│   └── ...                       # 539 total files
└── mask_buildings/
    ├── 15VF20_ws_building_mask.nc
    ├── 30VF20_ws_building_mask.nc
    └── ...
```

**NetCDF Schema**:
```python
# Wind speed file
xr.Dataset({
    'wind_speed': (['x', 'y', 'z'], float32),  # Shape: (288, 144, 130)
    # Metadata: simulation parameters
})

# Building mask file
xr.Dataset({
    'building_mask': (['x', 'y', 'z'], float32),  # Shape: (288, 144, 130)
    # Values: -1 (building), 1 (non-building)
})
```

### 3.2 WindSpeedDataset

#### Initialization

```python
dataset = WindSpeedDataset(
    wind_dir=Path("data/wind_speed_filled"),
    mask_dir=Path("data/mask_buildings"),
    target_shape=(512, 512, 64),  # (X, Y, Z)
    normalize=True,
    normalization_mode="per_sample",  # or "global"
    clip_range=(-1.0, 10.0),
    transform=None,  # Optional augmentation
    lazy_load=True   # Don't keep files open
)
```

#### Data Loading Pipeline

```
__getitem__(idx):
    1. Load raw arrays from NetCDF (with immediate file closure)
       wind_speed: (288, 144, 130) float32
       building_mask: (288, 144, 130) float32
    
    2. Pad or crop to target_shape
       - Z (height): Crop from BOTTOM (where buildings are)
       - X, Y: Crop from center or random
       → wind_speed: (512, 512, 64)
       → building_mask: (512, 512, 64)
    
    3. Clip wind speeds (optional)
       wind_speed = np.clip(wind_speed, -1.0, 10.0)
    
    4. Normalize (per-sample mode)
       non_building = wind_speed[building_mask != -1]
       mean = np.mean(non_building)
       std = np.std(non_building) + 1e-8
       wind_speed[non_building] = (wind_speed[non_building] - mean) / std
       wind_speed[building] = 0.0
    
    5. Extract and encode wind angle from filename
       filename: "15VF20_ws_filled.nc" → angle = 15°
       sin_angle, cos_angle = sin(15°), cos(15°)
    
    6. Create input tensor (building mask only)
       input: (1, 512, 512, 64) - building mask channel
    
    7. Create target tensor
       target: (1, 512, 512, 64) - normalized wind speed
    
    8. Apply optional transforms (augmentation)
    
    9. Return (input_tensor, target_tensor)
```

#### Normalization Modes

**Per-Sample Normalization** (default):
```python
# Compute statistics per sample
mask = building_mask != -1
mean = wind_speed[mask].mean()
std = wind_speed[mask].std() + 1e-8
normalized = (wind_speed - mean) / std
```

**Advantages**:
- Adapts to each sample's dynamic range
- No need to compute global stats upfront
- Better for heterogeneous scenarios

**Disadvantages**:
- Different normalization per sample
- Harder to interpret absolute values

**Global Normalization**:
```python
# Compute once at dataset init (sampling 30 files)
global_mean, global_std = compute_global_stats()

# Apply to all samples
normalized = (wind_speed - global_mean) / global_std
```

**Advantages**:
- Consistent normalization
- Preserves relative magnitudes

**Disadvantages**:
- Sensitive to outliers
- Expensive to compute for large datasets

### 3.3 Wind Angle Encoding

Wind direction is encoded as **sine and cosine** to handle circular discontinuity:

```python
def extract_angle_from_filename(filename):
    # "15VF20_ws_filled.nc" → 15
    match = re.search(r'^(\d+)', filename)
    return float(match.group(1)) if match else 0.0

def encode_angle(angle_degrees):
    angle_rad = np.radians(angle_degrees)
    return np.sin(angle_rad), np.cos(angle_rad)

# Example:
# 0° → (sin=0, cos=1)
# 90° → (sin=1, cos=0)
# 180° → (sin=0, cos=-1)
# 270° → (sin=-1, cos=0)
# 360° → (sin=0, cos=1)  # Same as 0°!
```

**Note**: Current implementation computes angle encoding but **doesn't concatenate to input tensor**. To use angle conditioning:

```python
# In dataset.__getitem__:
angle_sin_channel = np.full_like(building_mask, sin_angle)
angle_cos_channel = np.full_like(building_mask, cos_angle)
input_tensor = np.stack([building_mask, angle_sin_channel, angle_cos_channel])
```

And update model:
```python
model = UNet3D(in_channels=3, ...)  # Instead of 1
```

### 3.4 DataLoader Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    sampler=distributed_sampler,  # For multi-GPU
    shuffle=False if sampler else True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=3,        # Pre-load 3 batches per worker
    drop_last=True,           # Drop incomplete batches (for DDP)
    worker_init_fn=worker_init_fn,  # Per-worker seeding
    generator=torch.Generator().manual_seed(seed)
)
```

**Key Parameters**:

- **`num_workers`**: Number of parallel data loading processes
  - 0: Load in main process (simple, slow)
  - 4-8: Good balance for most systems
  - Too many: Overhead from IPC and RAM duplication

- **`pin_memory`**: Allocate CPU tensors in page-locked memory
  - Faster GPU transfer (CPU → GPU)
  - Uses more RAM (~2× batch size)
  - Essential for performance

- **`persistent_workers`**: Keep workers alive between epochs
  - Faster epoch transitions
  - Higher RAM usage
  - Recommended for large datasets

- **`prefetch_factor`**: Batches to pre-load per worker
  - Higher: Better GPU utilization, more RAM
  - Default: 2
  - Recommended: 3-4 for fast GPUs

- **`drop_last`**: Drop incomplete final batch
  - Required for DDP (all ranks must process same number of batches)
  - Slight data wastage (~1/world_size of data per epoch)

---


## 10. Data Augmentation

### 10.1 Physics-Safe Augmentations

**Constraints**:
- Wind flow is **anisotropic in Z** (vertical stratification)
- Buildings break symmetry
- Flips and rotations must preserve physics

**Allowed Augmentations**:
1. **Horizontal Flip (X or Y)**: ✓ Symmetric
2. **Horizontal Rotation (90°, 180°, 270°)**: ✓ Symmetric
3. **Gaussian Noise on Input**: ✓ Regularization

**Forbidden Augmentations**:
1. **Vertical Flip (Z)**: ✗ Breaks stratification
2. **Arbitrary Rotation**: ✗ Breaks vertical alignment
3. **Elastic Deformation**: ✗ Unrealistic geometry

### 10.2 Augmentation Classes

```python
from utils.augmentations import (
    RandomFlip3D,
    RandomRotate90_3D,
    AddGaussianNoise,
    Compose
)

transforms = Compose([
    RandomFlip3D(p_x=0.5, p_y=0.5),     # Flip X or Y with 50% prob
    RandomRotate90_3D(p=0.5),            # Rotate 90° in XY with 50% prob
    AddGaussianNoise(std=0.01, p=0.3)    # Add noise with 30% prob
])

dataset = WindSpeedDataset(..., transform=transforms)
```

### 10.3 Impact on Accuracy

**Empirical Results** (from config.py comments):
- No augmentation: Baseline
- Horizontal flips: +2-3% accuracy
- + Rotations: +3-5% accuracy
- + Noise: +1-2% accuracy (mild regularization)

**Total Expected Improvement**: +5-10%

---


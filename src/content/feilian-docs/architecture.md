## 1. Architecture Overview

### 1.1 System Design

The Feilian-3D system follows a modular architecture:

```
Input (Building Mask)
     ↓
Data Pipeline (utils/dataset.py)
     ├── Lazy Loading from NetCDF
     ├── Normalization (per-sample or global)
     ├── Padding/Cropping to Target Shape
     ├── Wind Angle Encoding (sin/cos)
     └── Optional Augmentation
     ↓
3D U-Net Model (models/unet3d.py)
     ├── Encoder (4 levels, MaxPool downsampling)
     ├── Bottleneck (16× base_channels)
     ├── Decoder (4 levels, interpolation upsampling)
     └── Skip Connections (concatenation)
     ↓
Loss Computation (utils/metrics.py)
     ├── Charbonnier Loss (robust MSE)
     └── Gradient Loss (physics constraint)
     ↓
Optimization
     ├── AdamW Optimizer
     ├── OneCycleLR Scheduler
     ├── Gradient Clipping
     └── Mixed Precision (AMP)
     ↓
Output (Wind Speed Field)
```

### 1.2 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Training Script** | `train.py` | Main entry point, orchestrates training loop |
| **Model** | `models/unet3d.py` | 3D U-Net architecture |
| **Dataset** | `utils/dataset.py` | Data loading and preprocessing |
| **Metrics** | `utils/metrics.py` | Loss functions and evaluation |
| **Distributed** | `utils/distributed.py` | Multi-GPU coordination |
| **Augmentation** | `utils/augmentations.py` | Data augmentation |
| **Config** | `config.py` | Centralized configuration |
| **SLURM** | `train_srun_robust.slm` | Job submission script |

### 1.3 Data Flow

**Training Step**:
1. `DistributedSampler` assigns batch indices to each GPU
2. `DataLoader` workers load and preprocess samples in parallel
3. Batch tensors are moved to GPU with `pin_memory` and `non_blocking=True`
4. Model forward pass with `channels_last_3d` memory format
5. Mixed precision (`torch.amp.autocast`) computes loss
6. Gradient computation and DDP synchronization
7. Optimizer step with gradient clipping
8. OneCycleLR scheduler step (per-batch, not per-epoch)
9. Metrics averaged across all GPUs with `dist.all_reduce`

---


## 2. 3D U-Net Model

### 2.1 Architecture

The model is a **3D U-Net** with modern improvements:

```
Input: (B, 1, X, Y, Z) = (batch, channels, width, height, depth)
Default: (1, 1, 512, 512, 64)

Encoder:
  enc1: ResConvBlock(1 → base_ch)             # (B, base_ch, 512, 512, 64)
    ↓ MaxPool3d(2) 
  enc2: ResConvBlock(base_ch → 2×base_ch)     # (B, 2×base_ch, 256, 256, 32)
    ↓ MaxPool3d(2)
  enc3: ResConvBlock(2×base_ch → 4×base_ch)   # (B, 4×base_ch, 128, 128, 16)
    ↓ MaxPool3d(2)
  enc4: ResConvBlock(4×base_ch → 8×base_ch)   # (B, 8×base_ch, 64, 64, 8)
    ↓ MaxPool3d(2)

Bottleneck:
  bottleneck: ResConvBlock(8×base_ch → 16×base_ch)  # (B, 16×base_ch, 32, 32, 4)

Decoder (with skip connections):
  up4: Interpolate + Conv1x1(16×base_ch → 8×base_ch)
  dec4: ResConvBlock(16×base_ch → 8×base_ch)  # Concat with enc4
  
  up3: Interpolate + Conv1x1(8×base_ch → 4×base_ch)
  dec3: ResConvBlock(8×base_ch → 4×base_ch)   # Concat with enc3
  
  up2: Interpolate + Conv1x1(4×base_ch → 2×base_ch)
  dec2: ResConvBlock(4×base_ch → 2×base_ch)   # Concat with enc2
  
  up1: Interpolate + Conv1x1(2×base_ch → base_ch)
  dec1: ResConvBlock(2×base_ch → base_ch)     # Concat with enc1

Output:
  out_conv: Conv3d(base_ch → 1)                # (B, 1, 512, 512, 64)
```

**Parameter Count**:
- `base_channels=16`: ~11M parameters
- `base_channels=32`: ~45M parameters
- `base_channels=64`: ~180M parameters

### 2.2 ResConvBlock Design

Each `ResConvBlock` contains:

```python
ResConvBlock(in_channels, out_channels):
    # First conv path
    Conv3d(in_channels → out_channels, kernel=3, padding=1, bias=False)
    GroupNorm(num_groups, out_channels)
    SiLU()
    
    # Second conv path
    Conv3d(out_channels → out_channels, kernel=3, padding=1, bias=False)
    GroupNorm(num_groups, out_channels)
    
    # Residual connection
    if in_channels != out_channels:
        skip = Conv3d(in_channels → out_channels, kernel=1, bias=False)
    else:
        skip = Identity()
    
    # Output
    output = conv_path + skip(input)
    output = SiLU()(output)
```

**Key Design Choices**:

1. **GroupNorm instead of BatchNorm**:
   - BatchNorm is unstable with `batch_size=1` (common for 3D volumes)
   - GroupNorm divides channels into groups (default 8)
   - If `out_channels % num_groups != 0`, automatically adjusts groups

2. **SiLU (Swish) instead of ReLU**:
   - Smoother activation: `SiLU(x) = x · sigmoid(x)`
   - Better gradient flow
   - No dead neurons (unlike ReLU)

3. **Residual Connections**:
   - Enables training deeper networks
   - Identity skip connection when `in_channels == out_channels`
   - 1×1 conv projection when channels change

4. **No Bias in Convolutions**:
   - GroupNorm has learnable affine parameters
   - Saves memory and parameters

### 2.3 Upsampling Strategy

**Phase 2 Improvement**: Uses `F.interpolate` + 1×1 conv instead of `ConvTranspose3d`

```python
# Old (Phase 1): ConvTranspose3d
up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

# New (Phase 2): Interpolate + Conv1x1
x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
x = nn.Conv3d(in_ch, out_ch, kernel_size=1)(x)
```

**Advantages**:
- No checkerboard artifacts
- More stable training
- Better gradient flow

### 2.4 Memory Optimizations

#### channels_last_3d Memory Format

```python
# Enable at model init
model = model.to(memory_format=torch.channels_last_3d)

# Apply to inputs during training
inputs = inputs.to(memory_format=torch.channels_last_3d)
```

**Benefits**:
- 5-15% performance improvement on modern GPUs
- Better cache locality for 3D convolutions
- Native support in ROCm MIOpen

**Memory Layout**:
- Standard (NCDHW): `[batch, channel, depth, height, width]`
- channels_last_3d (NDHWC): `[batch, depth, height, width, channel]`

#### Gradient Checkpointing

Optional memory saving (disabled by default):

```python
model = UNet3D(..., use_checkpointing=True)
```

- Recomputes activations during backward pass
- Trades compute for memory
- 30-40% memory reduction
- 10-20% slowdown

---


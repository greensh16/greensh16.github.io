## 12. API Reference

### 12.1 train.py

**Main Entry Point**:
```bash
python train.py [OPTIONS]
```

**Key Functions**:

#### `main(args)`
Orchestrates training pipeline.

**Steps**:
1. Initialize distributed training
2. Seed RNGs
3. Create datasets and dataloaders
4. Build model and wrap with DDP
5. Setup optimizer, scheduler, criterion
6. Resume from checkpoint (if specified)
7. Training loop
8. Cleanup

#### `train_epoch(model, loader, optimizer, criterion, grad_loss, device, scaler, use_amp, grad_weight, clip_grad, dist_info, epoch, scheduler)`
Trains for one epoch.

**Returns**: Dictionary of training metrics (loss, mse, grad_loss)

#### `validate(model, loader, criterion, device, hit_threshold, dist_info)`
Validates on validation set.

**Returns**: Dictionary of validation metrics (loss, mae, rmse, mre, hit_rate, nrmse)

#### `seed_everything(base_seed, rank)`
Seeds all RNGs for reproducibility.

**Returns**: Per-rank seed

### 12.2 models/unet3d.py

#### `UNet3D(in_channels=1, out_channels=1, base_channels=16, dropout=0.0, use_batchnorm=True, use_checkpointing=False)`
3D U-Net model.

**Parameters**:
- `in_channels`: Input channels (1 for building mask)
- `out_channels`: Output channels (1 for wind speed)
- `base_channels`: Base feature channels (16, 32, or 64)
- `dropout`: Dropout probability (not used in current version)
- `use_batchnorm`: Compatibility flag (ignored, always uses GroupNorm)
- `use_checkpointing`: Enable gradient checkpointing for memory savings

**Forward**:
- Input shape: `(B, in_channels, X, Y, Z)`
- Output shape: `(B, out_channels, X, Y, Z)`

#### `ResConvBlock(in_channels, out_channels, num_groups=8)`
Residual convolutional block.

**Architecture**: Conv → GroupNorm → SiLU → Conv → GroupNorm → Residual → SiLU

### 12.3 utils/dataset.py

#### `WindSpeedDataset(wind_dir, mask_dir, target_shape=(512, 512, 64), normalize=True, normalization_mode="per_sample", clip_range=(-1.0, 10.0), transform=None, ...)`
PyTorch Dataset for wind speed data.

**Returns**:
- `input_tensor`: `(1, X, Y, Z)` building mask
- `target_tensor`: `(1, X, Y, Z)` normalized wind speed

**Methods**:
- `__len__()`: Number of samples
- `__getitem__(idx)`: Load and preprocess sample
- `get_filename(idx)`: Get base filename

#### `create_dataloaders(wind_dir, mask_dir, batch_size=1, val_split=0.2, num_workers=4, pin_memory=True, ...)`
Create train and validation dataloaders.

**Returns**: `(train_loader, val_loader)`

### 12.4 utils/metrics.py

#### `CharbonnierLoss(epsilon=1e-3)`
Smooth L1 loss variant.

**Forward**:
- Input: `(pred, target)` tensors of any shape
- Output: Scalar loss

#### `ImprovedGradientLoss(weight_by_magnitude=True)`
Physics-informed gradient matching loss.

**Forward**:
- Input: `(pred, target)` tensors of shape `(B, C, X, Y, Z)`
- Output: Scalar loss (average of X, Y, Z gradients)

#### `compute_all_metrics(pred, target, building_mask=None, hit_threshold=0.1)`
Compute all evaluation metrics.

**Returns**: Dictionary with keys `mae`, `rmse`, `mre`, `hit_rate`, `nrmse`

### 12.5 utils/distributed.py

#### `setup_distributed_training()`
Initialize DDP from environment variables.

**Returns**: `DistributedInfo(rank, local_rank, world_size, device)` or `None`

#### `wrap_model_for_ddp(model, distributed_info)`
Wrap model with DistributedDataParallel.

**Returns**: DDP-wrapped model or original model

#### `create_distributed_sampler(dataset, distributed_info, shuffle=True)`
Create DistributedSampler for data distribution.

**Returns**: `DistributedSampler` or `None`

#### `sync_metrics(metrics, distributed_info)`
Average metrics across all ranks.

**Returns**: Dictionary of synchronized metrics

#### `is_main_rank(distributed_info)`
Check if current process is rank 0.

**Returns**: Boolean

#### `cleanup_distributed()`
Clean up distributed resources.

### 12.6 utils/augmentations.py

#### `RandomFlip3D(p_x=0.5, p_y=0.5)`
Random horizontal flips.

#### `RandomRotate90_3D(p=0.5)`
Random 90° rotations in XY plane.

#### `AddGaussianNoise(mean=0.0, std=0.01, p=0.3)`
Add Gaussian noise to input.

#### `Compose(transforms)`
Compose multiple transforms.

#### `get_training_augmentation(flip_x=True, flip_y=True, rotate_90=True, add_noise=False, ...)`
Get recommended augmentation pipeline.

**Returns**: `Compose` object or `None`

---

## Appendices

### A. Glossary

| Term | Definition |
|------|------------|
| **AMP** | Automatic Mixed Precision - FP16/FP32 hybrid training |
| **CFD** | Computational Fluid Dynamics |
| **DDP** | DistributedDataParallel - PyTorch multi-GPU framework |
| **MIOpen** | AMD's GPU-accelerated deep learning library (like cuDNN) |
| **NCCL** | NVIDIA Collective Communications Library (RCCL on AMD) |
| **ROCm** | AMD's GPU compute platform (like CUDA) |
| **Setonix** | Pawsey Supercomputing Centre's supercomputer |
| **SLURM** | Simple Linux Utility for Resource Management |
| **U-Net** | Encoder-decoder CNN architecture with skip connections |

### B. Hardware Specifications

**Setonix GPU Nodes**:
- **GPU**: AMD MI250X (8 per node)
- **GPU Memory**: 64GB HBM2e per GPU
- **CPU**: AMD EPYC 7A53 (64 cores per node)
- **RAM**: 512GB DDR4
- **Interconnect**: Slingshot 11 (200 Gb/s)
- **ROCm Version**: 6.3.3
- **PyTorch**: 2.7.1+rocm6.3.3

### C. Typical Training Timeline

**Single GPU (512×512×64, base_channels=32, batch_size=1)**:
- Iteration time: ~0.4s
- Batches per epoch: ~430 (539 samples × 0.8 / 1)
- Epoch time: ~3 minutes
- 30 epochs: ~1.5 hours

**16 GPUs (2 nodes)**:
- Effective batch size: 16
- Iteration time: ~0.5s (slight overhead)
- Batches per epoch: ~27 (430 / 16)
- Epoch time: ~15 seconds
- 30 epochs: ~8 minutes

**Scaling Efficiency**: 1.5 hours / (8 minutes × 2) ≈ 5.6× on 16 GPUs → 35% efficiency (typical for small batches)

### D. File Size Reference

| Component | Count | Size per Item | Total Size |
|-----------|-------|---------------|------------|
| Wind speed files | 539 | 50-320 MB | ~48 GB |
| Building mask files | 539 | 250-600 KB | ~100 MB |
| Model checkpoint | 1 | 63 MB | 63 MB |
| TensorBoard logs | Varies | Depends on epochs | ~10-50 MB |

---

**End of Documentation**

For questions or issues, please open a GitHub issue or contact the maintainers.


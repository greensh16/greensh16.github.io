## 8. Performance Optimizations

### 8.1 Summary of Optimizations

| Optimization | Speedup | Memory Saving | Implementation |
|--------------|---------|---------------|----------------|
| Mixed Precision (AMP) | 1.5-2× | 30-40% | `torch.amp.autocast` |
| channels_last_3d | 5-15% | None | `to(memory_format=...)` |
| MIOpen Benchmarking | 10-30% | None | `MIOPEN_FIND_MODE` |
| OneCycleLR | 1.5-2× (convergence) | None | `OneCycleLR` scheduler |
| Persistent Workers | 5-10% | Moderate | `persistent_workers=True` |
| Gradient Checkpointing | None | 30-40% | `use_checkpointing=True` |
| DDP Static Graph | 5-10% | None | `static_graph=True` |

### 8.2 Mixed Precision Details

**FP16 vs FP32 Operations**:

| Operation | Precision | Reason |
|-----------|-----------|--------|
| Conv3d | FP16 | Compute-intensive, tolerant |
| Linear | FP16 | Matrix multiply |
| GroupNorm | FP32 | Numerical stability |
| Loss | FP32 | Avoid underflow |
| Gradients (internal) | FP16 | Memory |
| Gradients (optimizer) | FP32 | Accumulation stability |

**ROCm AMP Considerations**:
- MI250X supports FP16 natively
- bfloat16 not widely supported (use FP16)
- Some ops slower in FP16 on ROCm (ProfilerActivity can identify)

### 8.3 Memory Format

**channels_last vs contiguous**:
```python
# Contiguous (NCDHW): [N, C, D, H, W]
# Memory layout: N₀C₀D₀H₀W₀, N₀C₀D₀H₀W₁, ..., N₀C₀D₀H₁W₀, ...

# channels_last_3d (NDHWC): [N, D, H, W, C]
# Memory layout: N₀D₀H₀W₀C₀, N₀D₀H₀W₀C₁, ..., N₀D₀H₀W₁C₀, ...
```

**Why Faster?**:
- 3D convolution accesses channels in inner loop
- channels_last_3d has better cache locality
- Fewer cache misses → faster compute

**When to Use**:
- ✓ 3D CNNs with many channels
- ✓ Modern GPUs (ROCm 5.0+, CUDA 11+)
- ✗ CPU inference (may be slower)
- ✗ Custom CUDA kernels (may not support)

### 8.4 DataLoader Tuning

**Guidelines**:
```python
# For fast GPUs (MI250X, A100)
num_workers = 4-8
pin_memory = True
persistent_workers = True
prefetch_factor = 3

# For slower GPUs (V100, T4)
num_workers = 2-4
pin_memory = True
persistent_workers = False
prefetch_factor = 2

# For CPU inference
num_workers = 0
pin_memory = False
```

**Profiling**:
```python
import time

t0 = time.time()
for batch in train_loader:
    t1 = time.time()
    print(f"Data loading: {t1 - t0:.3f}s")
    
    # Training step
    ...
    
    t2 = time.time()
    print(f"Training: {t2 - t1:.3f}s")
    t0 = t2
```

**Goal**: Data loading time < Training time

### 8.5 Compute vs Memory Tradeoffs

| Configuration | Compute | Memory | Use Case |
|---------------|---------|--------|----------|
| base_channels=64, no checkpointing | Fastest | Highest | A100 80GB |
| base_channels=32, no checkpointing | Fast | High | A100 40GB, MI250X |
| base_channels=16, no checkpointing | Medium | Medium | V100 32GB |
| base_channels=32, checkpointing | Slow | Medium | V100 32GB |
| base_channels=16, checkpointing | Slower | Low | V100 16GB |

---


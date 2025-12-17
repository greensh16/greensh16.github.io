## 5. Distributed Training

### 5.1 PyTorch Distributed Data Parallel (DDP)

**Architecture**:
```
Node 1                          Node 2
├── GPU 0 (Rank 0, Local 0)    ├── GPU 0 (Rank 8, Local 0)
├── GPU 1 (Rank 1, Local 1)    ├── GPU 1 (Rank 9, Local 1)
├── ...                         ├── ...
└── GPU 7 (Rank 7, Local 7)    └── GPU 7 (Rank 15, Local 7)
         ↓                               ↓
    Master Process (Rank 0)
    MASTER_ADDR=node1
    MASTER_PORT=29500
```

**Key Concepts**:

- **World Size**: Total number of processes (GPUs) = nodes × GPUs/node
  - Example: 2 nodes × 8 GPUs = 16

- **Rank**: Global process ID ∈ [0, world_size)
  - Rank 0 is "main" process (logging, checkpointing)

- **Local Rank**: Process ID within a node ∈ [0, GPUs/node)
  - Maps to GPU device index: `torch.cuda.set_device(local_rank)`

### 5.2 Initialization

```python
def setup_distributed_training():
    # Read environment variables set by SLURM/torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Initialize process group (NCCL for GPU communication)
    dist.init_process_group(
        backend="nccl",           # RCCL on AMD GPUs
        init_method="env://",     # Use environment variables
        world_size=world_size,
        rank=rank
    )
    
    # Synchronize all processes
    dist.barrier()
    
    return DistributedInfo(rank, local_rank, world_size, device)
```

**Environment Variables** (set by SLURM `srun`):
- `RANK`: Global rank
- `LOCAL_RANK`: Local rank
- `WORLD_SIZE`: Total processes
- `MASTER_ADDR`: Hostname of rank 0
- `MASTER_PORT`: Communication port (default 29500)

### 5.3 Model Wrapping

```python
model = UNet3D(...).to(device)

## 6. Training Pipeline

### 6.1 Training Loop Structure

```python
for epoch in range(start_epoch, epochs + 1):
    # 1. Training phase
    model.train()
    train_sampler.set_epoch(epoch)  # Shuffle for this epoch
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Convert to channels_last_3d
        inputs = inputs.to(memory_format=torch.channels_last_3d)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss_mse = criterion(outputs, targets)
            
            if grad_loss:
                loss_grad = grad_loss(outputs, targets)
                loss = loss_mse + gradient_weight * loss_grad
            else:
                loss = loss_mse
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first!)
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler (per-batch for OneCycleLR)
        if scheduler:
            scheduler.step()
        
        # Accumulate metrics
        running_loss += loss.item()
    
    # Synchronize training metrics
    train_metrics = sync_metrics(train_metrics, dist_info)
    
    # 2. Validation phase
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            # Compute metrics...
    
    val_metrics = sync_metrics(val_metrics, dist_info)
    
    # 3. Checkpointing (rank 0 only)
    if is_main_rank(dist_info) and val_metrics['loss'] < best_val_loss:
        save_checkpoint(...)
    
    # 4. Early stopping
    if early_stopping and patience_exceeded:
        break
```

### 6.2 Mixed Precision Training (AMP)

```python
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

with torch.amp.autocast('cuda', enabled=use_amp):
    outputs = model(inputs)  # FP16 ops
    loss = criterion(outputs, targets)  # FP32 loss

scaler.scale(loss).backward()  # Scale to prevent underflow
scaler.step(optimizer)         # Unscale gradients, then step
scaler.update()                # Adjust scale factor
```

**How AMP Works**:
1. **Autocasting**: Ops run in FP16 or FP32 depending on op type
   - Conv, Linear: FP16 (fast)
   - Loss, Norm: FP32 (stable)

2. **Loss Scaling**: Multiply loss by scale factor (e.g., 65536)
   - Prevents gradient underflow (FP16 range: 6e-5 to 65504)
   - Scaled gradients are representable in FP16

3. **Gradient Unscaling**: Divide gradients by scale factor before optimizer step

4. **Dynamic Scaling**: Adjusts scale factor if overflow/underflow detected

**Benefits**:
- 1.5-2× speedup on modern GPUs
- 30-40% memory reduction (smaller activations)
- Minimal accuracy loss (< 0.1% typically)

**When to Disable**:
- Numerical instability (NaNs in loss)
- Debugging (easier to isolate issues)
- Small models (overhead > benefit)

### 6.3 Gradient Clipping

```python
if clip_grad > 0:
    scaler.unscale_(optimizer)  # Must unscale first!
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
```

**Purpose**: Prevent gradient explosion

**How It Works**:
```
total_norm = √(Σ ||grad_i||²)
if total_norm > clip_grad:
    for param in params:
        param.grad *= clip_grad / total_norm
```

**Typical Values**:
- 0.0: No clipping
- 1.0 (default): Conservative
- 5.0: Permissive
- 10.0+: Rarely needed

**Signs You Need It**:
- Loss spikes to inf/NaN
- Gradient norms > 100 (log with `total_norm = clip_grad_norm_(...)`)

### 6.4 Learning Rate Scheduling

#### OneCycleLR

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.3,        # 30% warmup
    anneal_strategy='cos',
    div_factor=25.0,      # initial_lr = max_lr / 25
    final_div_factor=10000.0  # min_lr = initial_lr / 10000
)


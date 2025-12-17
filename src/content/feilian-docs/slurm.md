## 7. SLURM Configuration

### 7.1 train_srun_robust.slm

```bash
#!/bin/bash -l
#SBATCH --job-name=feilian3d
#SBATCH --partition=gpu              # GPU partition
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --ntasks-per-node=8          # 8 tasks (GPUs) per node
#SBATCH --gpus-per-node=8            # 8 GPUs per node
#SBATCH --time=08:00:00              # 8 hour time limit
#SBATCH --account=pawsey0928-gpu     # Account
#SBATCH --output=logs/train-%j.out   # Stdout log
#SBATCH --error=logs/train-%j.err    # Stderr log
```

**Resource Calculation**:
- Total GPUs = nodes × gpus-per-node = 2 × 8 = 16
- Total tasks (processes) = ntasks-per-node × nodes = 16
- Effective batch size = batch_size × num_GPUs = 1 × 16 = 16

### 7.2 Module Loading

```bash
module load pytorch/2.7.1-rocm6.3.3
export SINGULARITYENV_LD_PRELOAD=/opt/cray/libfabric/1.15.2.0/lib64/libfabric.so:$SINGULARITYENV_LD_PRELOAD
export SINGULARITYENV_PYTHONPATH=/software/projects/.../uptake-gpu/lib/python3.12/site-packages:$SINGULARITYENV_PYTHONPATH
```

**Purpose**:
- Load PyTorch container with ROCm support
- Inject libfabric for fast inter-node communication
- Add virtual environment to Python path

### 7.3 ROCm Optimizations

```bash
# MIOpen (AMD's cuDNN equivalent) settings
export MIOPEN_FIND_MODE=NORMAL         # Enable kernel search
export MIOPEN_DEBUG_DISABLE_FIND_DB=0  # Use find database
export MIOPEN_FIND_ENFORCE=3           # Enforce search
export MIOPEN_DISABLE_CACHE=0          # Enable cache
export PYTORCH_MIOPEN_SUGGEST_NHWC=0   # Don't force NHWC (we use channels_last_3d)
```

**Explanation**:
- MIOpen benchmarks kernels at runtime (like cuDNN)
- First run is slow (searching), subsequent runs are fast (cached)
- Find database persists across runs
- `FIND_ENFORCE=3`: Always search for best kernel

### 7.4 Distributed Environment

```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
```

**Variables**:
- `MASTER_ADDR`: Hostname of rank 0 (first node)
- `MASTER_PORT`: Communication port
- `WORLD_SIZE`: Total processes
- `NCCL_DEBUG=INFO`: Enable logging
- `NCCL_IB_DISABLE=1`: Disable InfiniBand (use Ethernet)
- `NCCL_SOCKET_IFNAME`: Exclude docker and loopback interfaces

### 7.5 srun Execution

```bash
srun --export=ALL \
     -N $SLURM_NNODES \
     -n $SLURM_NTASKS \
     -c 8 \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --gpus-per-node=$SLURM_GPUS_PER_NODE \
     --cpu-bind=cores \
     bash -c '
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        
        source /path/to/venv/bin/activate
        
        python train.py \
            --epochs 30 \
            --batch-size 1 \
            --base-channels 16 \
            --lr 0.001 \
            --use-amp \
            --use-gradient-loss \
            --gradient-weight 0.1 \
            --use-scheduler \
            --early-stopping \
            --patience 20 \
            --checkpoint-dir checkpoints \
            --log-dir logs
     '
```

**Key Flags**:
- `--export=ALL`: Pass all environment variables to tasks
- `-c 8`: 8 CPU cores per task (for data loading)
- `--cpu-bind=cores`: Bind tasks to specific cores (NUMA optimization)
- `bash -c '...'`: Each task runs this command

**Environment Mapping**:
- `SLURM_PROCID` → `RANK` (global rank)
- `SLURM_LOCALID` → `LOCAL_RANK` (local rank)

---


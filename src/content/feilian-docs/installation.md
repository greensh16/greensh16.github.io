# Installation & Environment Setup

## Prerequisites

- **Operating System**: Linux (tested on SLES on Setonix)
- **Python**: 3.9+
- **GPU**: AMD MI250X with ROCm 6.3.3 (or NVIDIA with CUDA 11+)
- **Storage**: ~50GB for data, ~5GB for checkpoints

## On Setonix (Pawsey Supercomputing Centre)

### 1. Load PyTorch Module

```bash
module load pytorch/2.7.1-rocm6.3.3
```

This loads:
- PyTorch 2.7.1 with ROCm 6.3.3 support
- Singularity container with Python 3.12
- ROCm GPU libraries (MIOpen, RCCL, etc.)

### 2. Create Virtual Environment

```bash
# Create virtual environment inside the container
module load singularity/4.1.0-mpi-gpu

# The container automatically provides Python
# Create venv in your project directory
python -m venv /software/projects/pawsey0928/$USER/feilian-gpu/uptake-gpu

# Activate
source /software/projects/pawsey0928/$USER/feilian-gpu/uptake-gpu/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
torch>=2.0.0          # Already provided by module
numpy>=1.24.0
xarray>=2023.1.0
h5netcdf>=1.1.0
netCDF4>=1.6.2
tqdm>=4.65.0
tensorboard>=2.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

### 4. Verify Installation

```bash
python << 'PYEOF'
import torch
import xarray as xr
import h5netcdf

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ROCm version: {torch.version.hip}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Test GPU
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    print(f"✓ GPU test passed: {x.device}")
else:
    print("✗ No GPU available")
PYEOF
```

Expected output:
```
PyTorch version: 2.7.1+rocm6.3.3
CUDA available: True
ROCm version: 6.3.3
Number of GPUs: 8
✓ GPU test passed: cuda:0
```

### 5. Configure Environment Variables

Add to your `~/.bashrc` or job script:

```bash
# MIOpen optimization
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_DISABLE_CACHE=0
export PYTORCH_MIOPEN_SUGGEST_NHWC=0

# Target shape (optional, can override in train.py)
export TARGET_SHAPE_Z=64
export TARGET_SHAPE_Y=512
export TARGET_SHAPE_X=512
```

## On Generic Linux with NVIDIA GPUs

### 1. Install PyTorch

```bash
# Create conda environment
conda create -n feilian python=3.10
conda activate feilian

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
PYEOF
```

## Data Setup

### 1. Organize Data Directory

```bash
cd /software/projects/pawsey0928/sgreen/feilian-3d

# Verify data structure
ls data/wind_speed_filled/*.nc | wc -l  # Should show 539
ls data/mask_buildings/*.nc | wc -l     # Should show 539
```

Expected structure:
```
data/
├── wind_speed_filled/
│   ├── 15VF20_ws_filled.nc
│   ├── 30VF20_ws_filled.nc
│   └── ... (539 files total)
└── mask_buildings/
    ├── 15VF20_ws_building_mask.nc
    ├── 30VF20_ws_building_mask.nc
    └── ... (539 files)
```

### 2. Verify Data Format

```bash
python << 'PYEOF'
import xarray as xr
from pathlib import Path

# Check one wind speed file
wind_file = Path("data/wind_speed_filled/15VF20_ws_filled.nc")
ds = xr.open_dataset(wind_file, engine='h5netcdf')
print(f"Wind speed shape: {ds['wind_speed'].shape}")
print(f"Wind speed dtype: {ds['wind_speed'].dtype}")
print(f"Variables: {list(ds.keys())}")

# Check one mask file
mask_file = Path("data/mask_buildings/15VF20_ws_building_mask.nc")
ds = xr.open_dataset(mask_file, engine='h5netcdf')
print(f"Mask shape: {ds['building_mask'].shape}")
print(f"Mask values: {ds['building_mask'].values.min()}, {ds['building_mask'].values.max()}")
PYEOF
```

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure virtual environment is activated:
```bash
which python  # Should point to venv
pip list | grep torch
```

### Issue: "No module named 'h5netcdf'"

**Solution**: 
```bash
pip install h5netcdf
```

### Issue: GPU not detected

**On Setonix**:
```bash
# Verify ROCm
rocm-smi

# Check GPU visibility
echo $ROCR_VISIBLE_DEVICES
```

**On NVIDIA**:
```bash
nvidia-smi
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or model capacity:
```bash
python train.py --batch-size 1 --base-channels 16
```

### Issue: MIOpen kernel search takes forever

**First run behavior**: MIOpen searches for optimal kernels (10-30 minutes)

**Subsequent runs**: Uses cached kernels (fast startup)

**To skip**: Set `MIOPEN_FIND_MODE=1` (use default kernels, no search)

### Issue: Permission denied on data files

**Solution**:
```bash
# Check permissions
ls -l data/wind_speed_filled/ | head

# Fix if needed (as owner)
chmod -R u+r data/
```

## Performance Validation

Run a quick training test:

```bash
# Single epoch, small subset
python train.py \
    --epochs 1 \
    --batch-size 1 \
    --base-channels 16 \
    --num-workers 2 \
    --val-split 0.1
```

Expected output:
```
Loaded 539 file pairs
Target shape: (512, 512, 64)
Model parameters: 11,234,567
✓ Enabled MIOpen benchmarking for AMD ROCm
Train samples: 485, Val samples: 54
Training...
Epoch 1/1: 100%|████████| 485/485 [03:45<00:00,  2.15it/s]
Train Loss: 0.4523 | Val Loss: 0.4102, MAE: 0.312
✓ Saved best model
```

## Next Steps

- [Quickstart Guide](Quickstart) - Run your first training
- [Configuration](Configuration-System) - Customize training settings
- [SLURM Guide](SLURM-Configuration) - Submit multi-GPU jobs on Setonix

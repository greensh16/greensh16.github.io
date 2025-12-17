## 4. Loss Functions & Metrics

### 4.1 Charbonnier Loss

**Definition**:
```python
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff**2 + self.epsilon**2)
        return loss.mean()
```

**Mathematical Form**:
```
L_char = √(|pred - target|² + ε²)
```

**Properties**:
- **Smooth L1 variant**: Differentiable at zero
- **Robust to outliers**: Less sensitive than MSE to large errors
- **Convex**: Global minimum exists
- **ε controls behavior**:
  - Small ε (1e-3): Nearly L1 for large errors
  - Large ε (1e-1): Nearly L2 (MSE) everywhere

**Comparison to Other Losses**:

| Loss | Formula | Outlier Sensitivity | Gradient at Zero |
|------|---------|---------------------|------------------|
| MSE | `(x)²` | High (quadratic) | 0 |
| L1 | `|x|` | Low (linear) | Undefined |
| Huber | Piecewise | Medium | Smooth |
| Charbonnier | `√(x² + ε²)` | Low-Medium | Smooth |

**Why Charbonnier?**:
- CFD data may have localized high-error regions (e.g., near building edges)
- MSE over-penalizes these regions
- Charbonnier balances accuracy and robustness

### 4.2 Improved Gradient Loss

**Purpose**: Enforce **spatial smoothness** and **physical plausibility**

**Implementation**:
```python
class ImprovedGradientLoss(nn.Module):
    def __init__(self, weight_by_magnitude=True):
        super().__init__()
        self.weight_by_magnitude = weight_by_magnitude
    
    def forward(self, pred, target):
        # Convert to FP32 for numerical stability
        pred = pred.float()
        target = target.float()
        
        # Compute finite difference gradients
        pred_dx = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_dx = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        
        pred_dz = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_dz = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        if self.weight_by_magnitude:
            # Weight by inverse gradient magnitude
            # → Focus on edges, not uniform regions
            weight_x = 1.0 / (target_dx.abs() + 1e-3)
            weight_y = 1.0 / (target_dy.abs() + 1e-3)
            weight_z = 1.0 / (target_dz.abs() + 1e-3)
            
            loss_x = ((pred_dx - target_dx).abs() * weight_x).mean()
            loss_y = ((pred_dy - target_dy).abs() * weight_y).mean()
            loss_z = ((pred_dz - target_dz).abs() * weight_z).mean()
        else:
            loss_x = (pred_dx - target_dx).abs().mean()
            loss_y = (pred_dy - target_dy).abs().mean()
            loss_z = (pred_dz - target_dz).abs().mean()
        
        return (loss_x + loss_y + loss_z) / 3.0
```

**Physical Interpretation**:
- Wind fields obey **continuity equations** (∇·v = 0 for incompressible flow)
- Abrupt spatial changes are physically implausible
- Gradient loss encourages **smooth transitions**

**Magnitude Weighting**:
- Uniform regions (low gradient) are easy to predict
- Edges and transitions (high gradient) are hard
- Weighting focuses loss on challenging regions

**FP32 Computation**:
- Mixed precision (AMP) uses FP16 for speed
- Gradient differences can be numerically sensitive
- Computing in FP32 improves stability

**Combined Loss**:
```python
loss = loss_charbonnier + λ * loss_gradient
# Default: λ = 0.1
```

**Effect of λ**:
- λ = 0: Pure Charbonnier (point-wise accuracy)
- λ = 0.1 (default): Balanced
- λ = 1.0: Strong smoothness constraint (may over-smooth)

### 4.3 Evaluation Metrics

All metrics computed on **non-building regions only**:

```python
mask = building_mask != -1
pred_valid = pred[mask]
target_valid = target[mask]
```

#### Mean Absolute Error (MAE)

```python
MAE = |pred - target|.mean()
```

- **Units**: Same as wind speed (normalized)
- **Interpretation**: Average absolute deviation
- **Range**: [0, ∞), lower is better

#### Root Mean Squared Error (RMSE)

```python
RMSE = √((pred - target)²).mean()
```

- **Units**: Same as wind speed
- **Interpretation**: Standard deviation of errors
- **Sensitive to outliers**: Large errors heavily penalized

#### Mean Relative Error (MRE)

```python
MRE = (|pred - target| / (|target| + ε)).mean()
```

- **Units**: Dimensionless (ratio)
- **Interpretation**: Average percentage error
- **Range**: [0, ∞)
- **Caveat**: Undefined when target ≈ 0, hence ε = 1e-6

#### Hit Rate

```python
Hit_Rate = (|pred - target| ≤ threshold).mean()
```

- **Units**: Fraction in [0, 1]
- **Interpretation**: % of predictions within threshold
- **Default threshold**: 0.1 (in normalized units)
- **Similar to**: Accuracy in classification

#### Normalized RMSE (NRMSE)

```python
NRMSE = RMSE / (target.max() - target.min())
```

- **Units**: Dimensionless
- **Interpretation**: RMSE relative to target range
- **Useful for**: Comparing across different scales

---


## 11. Reproducibility

### 11.1 Seeding Strategy

```python
def seed_everything(base_seed, rank):
    seed = base_seed + rank  # Per-rank seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

# Per-rank seeding
rank = dist_info.rank if dist_info else 0
rank_seed = seed_everything(args.seed, rank)

# DataLoader seeding
worker_init_fn = make_worker_init_fn(rank_seed * 1000)
generator = torch.Generator().manual_seed(rank_seed)

train_loader = DataLoader(..., worker_init_fn=worker_init_fn, generator=generator)
```

**Why Per-Rank Seeding?**:
- Each rank sees different data order (DistributedSampler)
- Workers within each rank need unique seeds
- Rank 0 seed = base_seed
- Rank 1 seed = base_seed + 1
- Worker 0 on Rank 0 = rank_seed * 1000 = base_seed * 1000

### 11.2 Deterministic Operations

**Caveats**:
```python
# For full reproducibility
torch.backends.cudnn.benchmark = False  # Disable kernel search
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

**Tradeoff**:
- Reproducibility: Exact same results across runs
- Performance: 20-30% slower (no kernel optimization)

**Recommendation**:
- Development: Enable for debugging
- Production: Disable for speed

### 11.3 Hardware Differences

**AMD MI250X vs NVIDIA A100**:
- Different kernel implementations
- Floating-point rounding differs
- Results may differ by ~0.1-1%

**Mitigation**:
- Document hardware used
- Report mean ± std over multiple runs
- Ensure trends are consistent

---


# CUDA Vector Addition

Adds two vectors of 1024 elements in parallel on GPU.

```bash
nix run
```

**What it does:**
- Creates vectors A = [0, 1, 2, ...] and B = [0, 2, 4, ...]
- Computes C = A + B on GPU using 1024 parallel threads
- Verifies result and prints success message
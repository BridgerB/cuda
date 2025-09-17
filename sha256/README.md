# CUDA SHA-256 Example

This example demonstrates GPU-accelerated SHA-256 hashing using CUDA.

## Features

- Complete SHA-256 implementation in CUDA
- Parallel computation of multiple hashes
- Performance benchmarking
- Verification against CPU results

## Usage

```bash
# Build and run
nix run

# Or build for development
nix develop
nvcc -o sha256 sha256.cu -lcudart
./sha256
```

## What It Does

1. Takes a test string: "Hello CUDA SHA-256!"
2. Computes 1024 variations by appending different 4-byte indices
3. Performs SHA-256 on each variation in parallel
4. Measures performance in megahashes per second
5. Verifies correctness

## Output Example

```
Computing 1024 SHA-256 hashes on GPU...
Input: "Hello CUDA SHA-256!"

First 5 results:
Hash 0: a1b2c3d4e5f6...
Hash 1: f6e5d4c3b2a1...
...

Performance:
Time: 2.45 ms
Hashes/second: 418367
Megahashes/second: 0.42 MH/s
```

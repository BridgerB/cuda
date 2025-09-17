# Bitcoin CUDA Miner

High-performance GPU-accelerated Bitcoin mining implementation designed to
integrate with TypeScript mining pools.

## Features

- **GPU Acceleration**: Utilizes CUDA for massive parallel mining (1000x+
  speedup over CPU)
- **Bitcoin Protocol Compliance**: Correct double SHA-256, block header parsing,
  and target comparison
- **Integration Ready**: Command-line interface designed for subprocess
  integration with existing miners
- **Genesis Block Testing**: Validate implementation against known Bitcoin
  genesis block
- **Real-time Progress**: Hash rate reporting and progress monitoring

## File Structure

```
miner/
├── flake.nix              # Nix build configuration
├── main.cu                # CLI interface and main program
├── bitcoin_miner.cu       # Core mining implementation
├── block_header.h         # Bitcoin block header structures
├── sha256_double.h        # Double SHA-256 function declarations
├── target_compare.h       # 256-bit target comparison functions
└── README.md              # This file
```

## Usage

### Build and Test

```bash
# Build the miner
nix build

# Test with genesis block (should find nonce 2083236893)
nix run .#test
```

### Command Line Interface

```bash
./bitcoin_miner <block_header_hex> <nonce_start> <nonce_end> <target_hex>
```

**Arguments:**

- `block_header_hex`: 80-byte block header in hex (160 hex characters)
- `nonce_start`: Starting nonce value (decimal)
- `nonce_end`: Ending nonce value (decimal)
- `target_hex`: 32-byte difficulty target in hex (64 hex characters)

**Output:**

- `FOUND <nonce> <hash>` - If winning nonce found
- `EXHAUSTED <attempts>` - If nonce range exhausted
- `ERROR <message>` - If error occurred

### Example (Genesis Block)

```bash
./bitcoin_miner \
  0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d00000000 \
  2083230000 2083240000 \
  00000000ffff0000000000000000000000000000000000000000000000000000
```

Expected output:

```
FOUND 2083236893 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

## Integration with TypeScript Miner

Replace the CPU mining loop in your `worker.ts` with CUDA subprocess calls:

```typescript
// Instead of CPU mining loop:
const miningResult = await mineAttempt(currentTemplate, nonce);

// Use CUDA miner:
const cudaResult = await spawnCudaMiner(
  serializedBlockHeader, // From your serializeBlockHeader()
  nonceStart,
  nonceEnd,
  blockTemplate.target,
);
```

## Performance

**Expected Performance:**

- CPU Miner: ~10-50 KH/s
- CUDA Miner: ~50-500 MH/s
- **Improvement: 1000-10000x speedup**

**Scaling:**

- Automatically uses all available GPU cores
- Supports nonce ranges up to 4.3 billion per call
- Progress reporting every 10M attempts

## Technical Details

### Block Header Format

Bitcoin block headers are exactly 80 bytes:

- Version (4 bytes)
- Previous block hash (32 bytes)
- Merkle root (32 bytes)
- Timestamp (4 bytes)
- Difficulty bits (4 bytes)
- Nonce (4 bytes)

### Mining Process

1. Parse hex block header into structure
2. Launch thousands of GPU threads
3. Each thread tests different nonce values
4. Compute double SHA-256 of block header
5. Compare result with difficulty target
6. Return first winning nonce found

### Target Comparison

Bitcoin uses 256-bit difficulty targets. Hash must be numerically less than
target to be valid. Comparison is done in big-endian byte order.

## Development

### Building from Source

```bash
# Development shell
nix develop

# Manual compilation
nvcc -o bitcoin_miner main.cu bitcoin_miner.cu -lcudart -O3
```

### Testing

The genesis block test provides a known-good validation:

- Known nonce: 2083236893
- Known hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
- Difficulty: 1 (easiest possible)

### Optimization

- Optimized for modern CUDA architectures
- Memory coalescing for maximum bandwidth
- Atomic operations for thread-safe result reporting
- Early termination when solution found

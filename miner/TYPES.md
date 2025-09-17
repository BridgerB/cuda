# Bitcoin CUDA Miner - Type Definitions

This document describes the data structures, types, and interfaces used in the
CUDA Bitcoin miner implementation.

## Core Data Structures

### BlockHeader

```c
typedef struct {
    uint32_t version;           // Block version number
    uint8_t prev_block[32];     // Previous block hash (little-endian)
    uint8_t merkle_root[32];    // Merkle root hash (little-endian)
    uint32_t timestamp;         // Block timestamp (Unix time)
    uint32_t bits;              // Difficulty target (compact format)
    uint32_t nonce;             // Proof-of-work nonce
} __attribute__((packed)) BlockHeader;
```

**Size**: Exactly 80 bytes (Bitcoin protocol requirement) **Endianness**: All
multi-byte fields stored in little-endian format **Usage**: Represents the
complete Bitcoin block header structure

### MiningResult

```c
typedef struct {
    uint32_t nonce;             // Winning nonce value
    uint8_t hash[HASH_SIZE];    // Resulting block hash (big-endian)
    int found;                  // Boolean flag: 1 if valid solution found
} MiningResult;
```

**Purpose**: Stores the result of a successful mining operation **Hash Format**:
Big-endian for display compatibility with Bitcoin tools

## Constants

### Size Definitions

```c
#define BLOCK_HEADER_SIZE 80    // Bitcoin block header size in bytes
#define HASH_SIZE 32            // SHA-256 hash size in bytes
#define TARGET_SIZE 32          // Difficulty target size in bytes
```

### SHA-256 Constants

```c
// Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
#define SHA256_H0 0x6a09e667
#define SHA256_H1 0xbb67ae85
#define SHA256_H2 0x3c6ef372
#define SHA256_H3 0xa54ff53a
#define SHA256_H4 0x510e527f
#define SHA256_H5 0x9b05688c
#define SHA256_H6 0x1f83d9ab
#define SHA256_H7 0x5be0cd19

// Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
__constant__ uint32_t K[64] = { /* 64 constants */ };
```

## CUDA Device Functions

### Cryptographic Functions

```c
__device__ uint32_t rotr(uint32_t x, uint32_t n);
__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t sig0(uint32_t x);
__device__ uint32_t sig1(uint32_t x);
__device__ uint32_t delta0(uint32_t x);
__device__ uint32_t delta1(uint32_t x);
```

**Purpose**: SHA-256 mathematical operations **Scope**: GPU device functions
only

### Hash Operations

```c
__device__ void sha256_transform(uint32_t *state, const uint8_t *data);
__device__ void sha256_hash(const uint8_t *input, size_t len, uint8_t *output);
__device__ void double_sha256(const uint8_t *input, size_t len, uint8_t *output);
```

**sha256_transform**: Processes single 64-byte block **sha256_hash**: Complete
SHA-256 with padding **double_sha256**: Bitcoin-specific SHA-256(SHA-256(x))

### Block Header Operations

```c
__device__ void serialize_block_header(const BlockHeader* header, uint8_t* output);
__device__ int compare_256bit(const uint8_t* hash, const uint8_t* target);
__device__ void reverse_bytes(uint8_t* data, size_t len);
```

**serialize_block_header**: Converts BlockHeader struct to 80-byte array
**compare_256bit**: Returns 1 if hash < target, 0 otherwise **reverse_bytes**:
In-place byte order reversal for endianness conversion

## Host Functions

### Utility Functions

```c
__host__ int hex_to_bytes(const char* hex_string, uint8_t* bytes, size_t max_bytes);
__host__ void bytes_to_hex(const uint8_t* bytes, size_t len, char* hex_string);
__host__ int parse_block_header_hex(const char* hex_string, BlockHeader* header);
```

**Return Values**:

- `hex_to_bytes`: Returns number of bytes parsed, or -1 on error
- `parse_block_header_hex`: Returns 0 on success, -1 on error

## CUDA Kernel

### Main Mining Kernel

```c
__global__ void bitcoin_mine_kernel(
    const BlockHeader* header_template,    // Block header template (without nonce)
    const uint8_t* target,                 // 32-byte difficulty target
    uint32_t nonce_start,                  // Starting nonce value
    uint32_t nonce_end,                    // Ending nonce value (inclusive)
    MiningResult* result,                  // Output: mining result
    volatile int* found_flag               // Atomic flag for early termination
);
```

**Thread Organization**: Each thread tests one nonce value **Memory Access**:
Coalesced reads for optimal performance **Synchronization**: Atomic operations
prevent race conditions

## Command Line Interface

### Input Parameters

```c
// Command line argument structure (conceptual)
struct CLIArgs {
    char block_header_hex[161];    // 160 hex chars + null terminator
    uint32_t nonce_start;          // Starting nonce (decimal)
    uint32_t nonce_end;            // Ending nonce (decimal)
    char target_hex[65];           // 64 hex chars + null terminator
};
```

### Output Formats

```c
// Success output format
"FOUND <nonce> <hash>"
// Example: "FOUND 2083236893 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

// Exhaustion output format
"EXHAUSTED <attempts>"
// Example: "EXHAUSTED 800001"

// Error output format
"ERROR <message>"
// Example: "ERROR Block header must be exactly 160 hex characters"
```

## Memory Layout

### GPU Memory Allocation

```c
// Device memory pointers
BlockHeader* d_header;         // sizeof(BlockHeader) = 80 bytes
uint8_t* d_target;            // TARGET_SIZE = 32 bytes
MiningResult* d_result;       // sizeof(MiningResult) ≈ 40 bytes
int* d_found_flag;            // sizeof(int) = 4 bytes
```

### Thread Configuration

```c
// Kernel launch parameters
const int threads_per_block = 256;      // Optimal for most GPUs
const int max_blocks = 65535;           // CUDA grid size limit
uint64_t nonces_per_launch = threads_per_block * max_blocks;  // ~16.7M nonces
```

## Data Validation

### Input Constraints

- **Block header hex**: Must be exactly 160 characters (80 bytes)
- **Target hex**: Must be exactly 64 characters (32 bytes)
- **Nonce range**: nonce_start ≤ nonce_end ≤ 0xFFFFFFFF
- **Hex strings**: Must contain only valid hexadecimal characters [0-9a-fA-F]

### Bitcoin Protocol Compliance

- **Endianness**: Little-endian for block header fields, big-endian for hash
  comparison
- **Hash algorithm**: Double SHA-256 as specified in Bitcoin protocol
- **Target format**: 256-bit difficulty target in big-endian format
- **Nonce space**: 32-bit unsigned integer (0 to 4,294,967,295)

## Error Handling

### Error Codes

```c
// Function return values
#define SUCCESS 0
#define ERROR_INVALID_HEX -1
#define ERROR_WRONG_LENGTH -2
#define ERROR_CUDA_LAUNCH -3
#define ERROR_MEMORY_ALLOCATION -4
```

### CUDA Error States

- **Kernel launch failure**: Invalid grid/block dimensions or insufficient
  memory
- **Memory allocation failure**: Insufficient GPU memory
- **Synchronization timeout**: Kernel execution timeout (rare)

## Performance Characteristics

### Computational Complexity

- **Per-nonce operations**: O(1) - constant time hash computation
- **Memory bandwidth**: Limited by global memory access patterns
- **Scalability**: Linear with number of CUDA cores available

### Memory Access Patterns

- **Coalesced access**: Block header template read by all threads
- **Scattered writes**: Result written by single successful thread
- **Atomic operations**: Minimal contention on found_flag

This type system ensures Bitcoin protocol compliance while maximizing GPU
performance through optimized memory layouts and CUDA programming patterns.

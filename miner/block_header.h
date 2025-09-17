#ifndef BLOCK_HEADER_H
#define BLOCK_HEADER_H

#include <stdint.h>

#define BLOCK_HEADER_SIZE 80
#define HASH_SIZE 32
#define TARGET_SIZE 32

// Bitcoin block header structure (80 bytes total)
typedef struct {
    uint32_t version;           // 4 bytes - Block version
    uint8_t prev_block[32];     // 32 bytes - Previous block hash (little endian)
    uint8_t merkle_root[32];    // 32 bytes - Merkle root (little endian)
    uint32_t timestamp;         // 4 bytes - Block timestamp
    uint32_t bits;              // 4 bytes - Difficulty target (compact format)
    uint32_t nonce;             // 4 bytes - Nonce
} __attribute__((packed)) BlockHeader;

// Mining result structure
typedef struct {
    uint32_t nonce;
    uint8_t hash[HASH_SIZE];
    int found;
} MiningResult;

// Function declarations
__device__ void serialize_block_header(const BlockHeader* header, uint8_t* output);
__device__ int is_valid_hash(const uint8_t* hash, const uint8_t* target);
__host__ int hex_to_bytes(const char* hex_string, uint8_t* bytes, size_t max_bytes);
__host__ void bytes_to_hex(const uint8_t* bytes, size_t len, char* hex_string);
__host__ int parse_block_header_hex(const char* hex_string, BlockHeader* header);

#endif // BLOCK_HEADER_H
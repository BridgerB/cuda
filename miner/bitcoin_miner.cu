#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "block_header.h"
#include "sha256_double.h"
#include "target_compare.h"

// SHA-256 utility functions
__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t sig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t delta0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t delta1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void sha256_transform(uint32_t *state, const uint8_t *data) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        W[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | 
               (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    }
    
    for (int i = 16; i < 64; i++) {
        W[i] = delta1(W[i - 2]) + W[i - 7] + delta0(W[i - 15]) + W[i - 16];
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        t1 = h + sig1(e) + ch(e, f, g) + K[i] + W[i];
        t2 = sig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha256_hash(const uint8_t *input, size_t len, uint8_t *output) {
    uint32_t state[8] = {
        SHA256_H0, SHA256_H1, SHA256_H2, SHA256_H3,
        SHA256_H4, SHA256_H5, SHA256_H6, SHA256_H7
    };
    
    uint8_t buffer[64];
    size_t buffer_len = 0;
    size_t total_len = 0;
    
    // Process input
    while (len > 0) {
        size_t chunk = (len < (64 - buffer_len)) ? len : (64 - buffer_len);
        for (size_t i = 0; i < chunk; i++) {
            buffer[buffer_len + i] = input[total_len + i];
        }
        buffer_len += chunk;
        total_len += chunk;
        len -= chunk;
        
        if (buffer_len == 64) {
            sha256_transform(state, buffer);
            buffer_len = 0;
        }
    }
    
    // Padding
    buffer[buffer_len++] = 0x80;
    if (buffer_len > 56) {
        while (buffer_len < 64) buffer[buffer_len++] = 0;
        sha256_transform(state, buffer);
        buffer_len = 0;
    }
    while (buffer_len < 56) buffer[buffer_len++] = 0;
    
    // Append length
    uint64_t bit_len = total_len * 8;
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bit_len >> (56 - i * 8)) & 0xff;
    }
    sha256_transform(state, buffer);
    
    // Output hash
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (state[i] >> 24) & 0xff;
        output[i * 4 + 1] = (state[i] >> 16) & 0xff;
        output[i * 4 + 2] = (state[i] >> 8) & 0xff;
        output[i * 4 + 3] = state[i] & 0xff;
    }
}

__device__ void double_sha256(const uint8_t *input, size_t len, uint8_t *output) {
    uint8_t first_hash[32];
    sha256_hash(input, len, first_hash);
    sha256_hash(first_hash, 32, output);
}

// Target comparison functions
__device__ int compare_256bit(const uint8_t* hash, const uint8_t* target) {
    // Compare from most significant byte to least significant
    // Both hash and target should already be in big-endian format
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return 1;  // hash < target
        if (hash[i] > target[i]) return 0;  // hash > target
    }
    return 0;  // hash == target (not valid for mining)
}

__device__ void reverse_bytes(uint8_t* data, size_t len) {
    for (size_t i = 0; i < len / 2; i++) {
        uint8_t temp = data[i];
        data[i] = data[len - 1 - i];
        data[len - 1 - i] = temp;
    }
}

__device__ int count_leading_zero_bits(const uint8_t* hash) {
    int zeros = 0;
    for (int i = 31; i >= 0; i--) {  // Start from most significant byte
        if (hash[i] == 0) {
            zeros += 8;
        } else {
            // Count leading zeros in this byte
            uint8_t byte = hash[i];
            while ((byte & 0x80) == 0 && zeros < 256) {
                zeros++;
                byte <<= 1;
            }
            break;
        }
    }
    return zeros;
}

// Block header functions
__device__ void serialize_block_header(const BlockHeader* header, uint8_t* output) {
    int offset = 0;
    
    // Version (4 bytes, little endian)
    output[offset++] = header->version & 0xff;
    output[offset++] = (header->version >> 8) & 0xff;
    output[offset++] = (header->version >> 16) & 0xff;
    output[offset++] = (header->version >> 24) & 0xff;
    
    // Previous block hash (32 bytes, already little endian)
    for (int i = 0; i < 32; i++) {
        output[offset++] = header->prev_block[i];
    }
    
    // Merkle root (32 bytes, already little endian)
    for (int i = 0; i < 32; i++) {
        output[offset++] = header->merkle_root[i];
    }
    
    // Timestamp (4 bytes, little endian)
    output[offset++] = header->timestamp & 0xff;
    output[offset++] = (header->timestamp >> 8) & 0xff;
    output[offset++] = (header->timestamp >> 16) & 0xff;
    output[offset++] = (header->timestamp >> 24) & 0xff;
    
    // Bits (4 bytes, little endian)
    output[offset++] = header->bits & 0xff;
    output[offset++] = (header->bits >> 8) & 0xff;
    output[offset++] = (header->bits >> 16) & 0xff;
    output[offset++] = (header->bits >> 24) & 0xff;
    
    // Nonce (4 bytes, little endian)
    output[offset++] = header->nonce & 0xff;
    output[offset++] = (header->nonce >> 8) & 0xff;
    output[offset++] = (header->nonce >> 16) & 0xff;
    output[offset++] = (header->nonce >> 24) & 0xff;
}

// Main Bitcoin mining kernel
__global__ void bitcoin_mine_kernel(
    const BlockHeader* header_template,
    const uint8_t* target,
    uint32_t nonce_start,
    uint32_t nonce_end,
    MiningResult* result,
    volatile int* found_flag
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = nonce_start + idx;
    
    // Check if we're within range and no solution found yet
    if (nonce > nonce_end || *found_flag) return;
    
    // Create local copy of header with our nonce
    BlockHeader local_header = *header_template;
    local_header.nonce = nonce;
    
    // Serialize the block header
    uint8_t serialized_header[BLOCK_HEADER_SIZE];
    serialize_block_header(&local_header, serialized_header);
    
    // Compute double SHA-256
    uint8_t hash[HASH_SIZE];
    double_sha256(serialized_header, BLOCK_HEADER_SIZE, hash);
    
    // For debugging: if this is the known genesis nonce, print the hash
    if (nonce == 2083236893) {
        printf("CUDA DEBUG: Testing nonce %u\n", nonce);
        printf("CUDA DEBUG: Raw hash (little endian): ");
        for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
        printf("\n");
        
        // Reverse for big-endian comparison
        uint8_t reversed_hash[HASH_SIZE];
        for (int i = 0; i < HASH_SIZE; i++) {
            reversed_hash[i] = hash[31 - i];
        }
        
        printf("CUDA DEBUG: Hash (big endian): ");
        for (int i = 0; i < 32; i++) printf("%02x", reversed_hash[i]);
        printf("\n");
        
        printf("CUDA DEBUG: Target: ");
        for (int i = 0; i < 32; i++) printf("%02x", target[i]);
        printf("\n");
        
        // Test comparison
        int is_valid = compare_256bit(reversed_hash, target);
        printf("CUDA DEBUG: Hash < Target: %s\n", is_valid ? "YES" : "NO");
        
        // Also test with original unreversed hash
        int is_valid_unreversed = compare_256bit(hash, target);
        printf("CUDA DEBUG: Raw hash < Target: %s\n", is_valid_unreversed ? "YES" : "NO");
    }
    
    // Reverse hash for comparison (Bitcoin displays hashes in big-endian)
    reverse_bytes(hash, HASH_SIZE);
    
    // Check if hash meets target
    if (compare_256bit(hash, target)) {
        // Use atomic operation to ensure only first thread sets result
        if (atomicCAS((int*)found_flag, 0, 1) == 0) {
            result->nonce = nonce;
            result->found = 1;
            for (int i = 0; i < HASH_SIZE; i++) {
                result->hash[i] = hash[i];
            }
        }
    }
}

// Host utility functions
__host__ int hex_to_bytes(const char* hex_string, uint8_t* bytes, size_t max_bytes) {
    size_t hex_len = strlen(hex_string);
    if (hex_len % 2 != 0 || hex_len / 2 > max_bytes) {
        return -1;  // Invalid hex string or too long
    }
    
    for (size_t i = 0; i < hex_len; i += 2) {
        char hex_byte[3] = {hex_string[i], hex_string[i + 1], '\0'};
        bytes[i / 2] = (uint8_t)strtol(hex_byte, NULL, 16);
    }
    
    return hex_len / 2;  // Return number of bytes written
}

__host__ void bytes_to_hex(const uint8_t* bytes, size_t len, char* hex_string) {
    for (size_t i = 0; i < len; i++) {
        sprintf(hex_string + i * 2, "%02x", bytes[i]);
    }
    hex_string[len * 2] = '\0';
}

__host__ int parse_block_header_hex(const char* hex_string, BlockHeader* header) {
    uint8_t header_bytes[BLOCK_HEADER_SIZE];
    
    // Convert hex string to bytes
    int byte_count = hex_to_bytes(hex_string, header_bytes, BLOCK_HEADER_SIZE);
    if (byte_count != BLOCK_HEADER_SIZE) {
        return -1;  // Invalid header size
    }
    
    int offset = 0;
    
    // Parse version (4 bytes, little endian)
    header->version = header_bytes[offset] | 
                     (header_bytes[offset + 1] << 8) |
                     (header_bytes[offset + 2] << 16) |
                     (header_bytes[offset + 3] << 24);
    offset += 4;
    
    // Parse previous block hash (32 bytes)
    for (int i = 0; i < 32; i++) {
        header->prev_block[i] = header_bytes[offset + i];
    }
    offset += 32;
    
    // Parse merkle root (32 bytes)
    for (int i = 0; i < 32; i++) {
        header->merkle_root[i] = header_bytes[offset + i];
    }
    offset += 32;
    
    // Parse timestamp (4 bytes, little endian)
    header->timestamp = header_bytes[offset] |
                       (header_bytes[offset + 1] << 8) |
                       (header_bytes[offset + 2] << 16) |
                       (header_bytes[offset + 3] << 24);
    offset += 4;
    
    // Parse bits (4 bytes, little endian)
    header->bits = header_bytes[offset] |
                  (header_bytes[offset + 1] << 8) |
                  (header_bytes[offset + 2] << 16) |
                  (header_bytes[offset + 3] << 24);
    offset += 4;
    
    // Parse nonce (4 bytes, little endian) - will be overwritten during mining
    header->nonce = header_bytes[offset] |
                   (header_bytes[offset + 1] << 8) |
                   (header_bytes[offset + 2] << 16) |
                   (header_bytes[offset + 3] << 24);
    
    return 0;  // Success
}
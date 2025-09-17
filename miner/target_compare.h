#ifndef TARGET_COMPARE_H
#define TARGET_COMPARE_H

#include <stdint.h>

// Compare two 256-bit values represented as 32-byte arrays
// Returns 1 if hash < target, 0 otherwise
// Comparison is done in big-endian order (most significant bytes first)
__device__ int compare_256bit(const uint8_t* hash, const uint8_t* target);

// Reverse bytes in place (for endianness conversion)
__device__ void reverse_bytes(uint8_t* data, size_t len);

// Count leading zero bits in a hash (for difficulty estimation)
__device__ int count_leading_zero_bits(const uint8_t* hash);

#endif // TARGET_COMPARE_H
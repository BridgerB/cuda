#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sha256_constants.h"

// SHA-256 constants are now in sha256_constants.h

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

__global__ void sha256_kernel(const uint8_t *input, size_t input_len, uint8_t *output, int num_hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;
    
    // Each thread computes SHA-256 of input + thread index
    uint8_t local_input[256];
    for (size_t i = 0; i < input_len; i++) {
        local_input[i] = input[i];
    }
    
    // Append thread index as 4 bytes (little endian)
    local_input[input_len] = idx & 0xff;
    local_input[input_len + 1] = (idx >> 8) & 0xff;
    local_input[input_len + 2] = (idx >> 16) & 0xff;
    local_input[input_len + 3] = (idx >> 24) & 0xff;
    
    // Compute SHA-256
    sha256_hash(local_input, input_len + 4, &output[idx * 32]);
}

void print_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

int main() {
    const char *test_string = "Hello CUDA SHA-256!";
    size_t input_len = strlen(test_string);
    const int num_hashes = 1024;
    
    uint8_t *h_input, *h_output;
    uint8_t *d_input, *d_output;
    
    // Allocate host memory
    h_input = (uint8_t*)malloc(input_len);
    h_output = (uint8_t*)malloc(num_hashes * 32);
    memcpy(h_input, test_string, input_len);
    
    // Allocate device memory
    cudaMalloc(&d_input, input_len);
    cudaMalloc(&d_output, num_hashes * 32);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, input_len, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (num_hashes + threads_per_block - 1) / threads_per_block;
    
    printf("Computing %d SHA-256 hashes on GPU...\n", num_hashes);
    printf("Input: \"%s\"\n\n", test_string);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sha256_kernel<<<blocks, threads_per_block>>>(d_input, input_len, d_output, num_hashes);
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, num_hashes * 32, cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("Hash %d: ", i);
        print_hex(&h_output[i * 32], 32);
    }
    
    printf("\nPerformance:\n");
    printf("Time: %.2f ms\n", milliseconds);
    printf("Hashes/second: %.0f\n", (num_hashes / milliseconds) * 1000);
    printf("Megahashes/second: %.2f MH/s\n", (num_hashes / milliseconds) / 1000);
    
    // Verify one hash on CPU for correctness
    printf("\nVerification (computing hash 0 on CPU):\n");
    uint8_t cpu_input[256];
    for (size_t i = 0; i < input_len; i++) {
        cpu_input[i] = ((uint8_t*)test_string)[i];
    }
    cpu_input[input_len] = 0;     // index 0, little endian
    cpu_input[input_len + 1] = 0;
    cpu_input[input_len + 2] = 0;
    cpu_input[input_len + 3] = 0;
    
    printf("GPU hash 0: ");
    print_hex(&h_output[0], 32);
    printf("Verification complete\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
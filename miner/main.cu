#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "block_header.h"
#include "sha256_double.h"
#include "target_compare.h"

// External functions from bitcoin_miner.cu
__global__ void bitcoin_mine_kernel(
    const BlockHeader* header_template,
    const uint8_t* target,
    uint32_t nonce_start,
    uint32_t nonce_end,
    MiningResult* result,
    volatile int* found_flag
);

// These functions are declared in block_header.h and implemented in bitcoin_miner.cu

void print_usage(const char* program_name) {
    printf("Usage: %s <block_header_hex> <nonce_start> <nonce_end> <target_hex>\n", program_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  block_header_hex: 80-byte block header in hex format (160 hex characters)\n");
    printf("  nonce_start:      Starting nonce value (decimal)\n");
    printf("  nonce_end:        Ending nonce value (decimal)\n");
    printf("  target_hex:       32-byte difficulty target in hex format (64 hex characters)\n");
    printf("\n");
    printf("Example (Genesis block):\n");
    printf("  %s \\\n", program_name);
    printf("    0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d00000000 \\\n");
    printf("    0 4294967295 \\\n");
    printf("    00000000ffff0000000000000000000000000000000000000000000000000000\n");
    printf("\n");
    printf("Output:\n");
    printf("  If successful: FOUND <nonce> <hash>\n");
    printf("  If exhausted:  EXHAUSTED <attempts>\n");
    printf("  If error:      ERROR <message>\n");
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* header_hex = argv[1];
    uint32_t nonce_start = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t nonce_end = (uint32_t)strtoul(argv[3], NULL, 10);
    const char* target_hex = argv[4];
    
    // Debug: Print all inputs
    printf("DEBUG: Input header_hex: %s\n", header_hex);
    printf("DEBUG: Input nonce_start: %u\n", nonce_start);
    printf("DEBUG: Input nonce_end: %u\n", nonce_end);
    printf("DEBUG: Input target_hex: %s\n", target_hex);
    printf("DEBUG: Header hex length: %zu\n", strlen(header_hex));
    printf("DEBUG: Target hex length: %zu\n", strlen(target_hex));
    
    // Validate inputs
    if (strlen(header_hex) != 160) {
        printf("ERROR Block header must be exactly 160 hex characters (80 bytes)\n");
        return 1;
    }
    
    if (strlen(target_hex) != 64) {
        printf("ERROR Target must be exactly 64 hex characters (32 bytes)\n");
        return 1;
    }
    
    if (nonce_start > nonce_end) {
        printf("ERROR Nonce start must be <= nonce end\n");
        return 1;
    }
    
    // Parse block header
    BlockHeader header;
    if (parse_block_header_hex(header_hex, &header) != 0) {
        printf("ERROR Failed to parse block header hex\n");
        return 1;
    }
    
    // Debug: Print parsed header
    printf("DEBUG: Parsed header version: %u (0x%08x)\n", header.version, header.version);
    printf("DEBUG: Parsed header timestamp: %u (0x%08x)\n", header.timestamp, header.timestamp);
    printf("DEBUG: Parsed header bits: %u (0x%08x)\n", header.bits, header.bits);
    printf("DEBUG: Parsed header nonce: %u (0x%08x)\n", header.nonce, header.nonce);
    
    printf("DEBUG: Previous block hash: ");
    for (int i = 0; i < 32; i++) printf("%02x", header.prev_block[i]);
    printf("\n");
    
    printf("DEBUG: Merkle root: ");
    for (int i = 0; i < 32; i++) printf("%02x", header.merkle_root[i]);
    printf("\n");
    
    // Parse target
    uint8_t target[TARGET_SIZE];
    if (hex_to_bytes(target_hex, target, TARGET_SIZE) != TARGET_SIZE) {
        printf("ERROR Failed to parse target hex\n");
        return 1;
    }
    
    // Debug: Print parsed target
    printf("DEBUG: Parsed target: ");
    for (int i = 0; i < 32; i++) printf("%02x", target[i]);
    printf("\n");
    
    // Manual test of known nonce
    if (nonce_start == nonce_end && nonce_start == 2083236893) {
        printf("\nDEBUG: Testing known genesis nonce manually...\n");
        
        BlockHeader test_header = header;
        test_header.nonce = 2083236893;
        
        // Serialize manually for verification
        uint8_t serialized[80];
        
        // Version (little endian)
        serialized[0] = test_header.version & 0xff;
        serialized[1] = (test_header.version >> 8) & 0xff;
        serialized[2] = (test_header.version >> 16) & 0xff;
        serialized[3] = (test_header.version >> 24) & 0xff;
        
        // Previous block hash (32 bytes)
        for (int i = 0; i < 32; i++) {
            serialized[4 + i] = test_header.prev_block[i];
        }
        
        // Merkle root (32 bytes)
        for (int i = 0; i < 32; i++) {
            serialized[36 + i] = test_header.merkle_root[i];
        }
        
        // Timestamp (little endian)
        serialized[68] = test_header.timestamp & 0xff;
        serialized[69] = (test_header.timestamp >> 8) & 0xff;
        serialized[70] = (test_header.timestamp >> 16) & 0xff;
        serialized[71] = (test_header.timestamp >> 24) & 0xff;
        
        // Bits (little endian)
        serialized[72] = test_header.bits & 0xff;
        serialized[73] = (test_header.bits >> 8) & 0xff;
        serialized[74] = (test_header.bits >> 16) & 0xff;
        serialized[75] = (test_header.bits >> 24) & 0xff;
        
        // Nonce (little endian)
        serialized[76] = test_header.nonce & 0xff;
        serialized[77] = (test_header.nonce >> 8) & 0xff;
        serialized[78] = (test_header.nonce >> 16) & 0xff;
        serialized[79] = (test_header.nonce >> 24) & 0xff;
        
        printf("DEBUG: Serialized header with nonce %u: ", test_header.nonce);
        for (int i = 0; i < 80; i++) printf("%02x", serialized[i]);
        printf("\n");
        
        // Compute SHA-256 manually
        uint8_t first_hash[32];
        uint8_t final_hash[32];
        
        // First SHA-256
        // We need to implement this on CPU for debugging
        printf("DEBUG: Would compute double SHA-256 here...\n");
        printf("DEBUG: Expected final hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f\n");
    }
    
    // Calculate total nonces to test
    uint64_t total_nonces = (uint64_t)nonce_end - (uint64_t)nonce_start + 1;
    
    // GPU configuration
    const int threads_per_block = 256;
    const int max_blocks = 65535;  // GPU limit
    uint64_t nonces_per_launch = (uint64_t)threads_per_block * max_blocks;
    
    // Allocate device memory
    BlockHeader* d_header;
    uint8_t* d_target;
    MiningResult* d_result;
    int* d_found_flag;
    
    printf("DEBUG: Allocating GPU memory...\n");
    cudaMalloc(&d_header, sizeof(BlockHeader));
    cudaMalloc(&d_target, TARGET_SIZE);
    cudaMalloc(&d_result, sizeof(MiningResult));
    cudaMalloc(&d_found_flag, sizeof(int));
    
    // Copy data to device
    printf("DEBUG: Copying data to GPU...\n");
    cudaMemcpy(d_header, &header, sizeof(BlockHeader), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, TARGET_SIZE, cudaMemcpyHostToDevice);
    
    // Initialize result
    MiningResult host_result = {0, {0}, 0};
    cudaMemcpy(d_result, &host_result, sizeof(MiningResult), cudaMemcpyHostToDevice);
    
    printf("Mining range: %u to %u (%llu nonces)\n", nonce_start, nonce_end, total_nonces);
    
    // Mining loop
    uint32_t current_nonce = nonce_start;
    uint64_t total_attempts = 0;
    clock_t start_time = clock();
    
    while (current_nonce <= nonce_end) {
        // Calculate range for this launch
        uint32_t launch_end = current_nonce + nonces_per_launch - 1;
        if (launch_end > nonce_end) {
            launch_end = nonce_end;
        }
        
        uint32_t launch_nonces = launch_end - current_nonce + 1;
        int blocks = (launch_nonces + threads_per_block - 1) / threads_per_block;
        if (blocks > max_blocks) blocks = max_blocks;
        
        // Reset found flag
        int zero = 0;
        cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch kernel
        printf("DEBUG: Launching kernel with %d blocks, %d threads per block\n", blocks, threads_per_block);
        printf("DEBUG: Testing nonce range %u to %u (%u nonces)\n", current_nonce, launch_end, launch_nonces);
        
        bitcoin_mine_kernel<<<blocks, threads_per_block>>>(
            d_header, d_target, current_nonce, launch_end, d_result, d_found_flag
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
        
        printf("DEBUG: Kernel launched, waiting for completion...\n");
        cudaDeviceSynchronize();
        printf("DEBUG: Kernel completed\n");
        
        // Check if solution found
        int found_flag;
        cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        printf("DEBUG: Found flag value: %d\n", found_flag);
        
        if (found_flag) {
            // Copy result back
            cudaMemcpy(&host_result, d_result, sizeof(MiningResult), cudaMemcpyDeviceToHost);
            
            printf("DEBUG: Solution found! Nonce: %u\n", host_result.nonce);
            printf("DEBUG: Hash bytes: ");
            for (int i = 0; i < 32; i++) printf("%02x", host_result.hash[i]);
            printf("\n");
            
            // Convert hash to hex string
            char hash_hex[65];
            bytes_to_hex(host_result.hash, HASH_SIZE, hash_hex);
            
            printf("FOUND %u %s\n", host_result.nonce, hash_hex);
            goto cleanup;
        } else {
            printf("DEBUG: No solution found in this batch\n");
        }
        
        total_attempts += launch_nonces;
        current_nonce = launch_end + 1;
        
        // Progress report every 10 million attempts
        if (total_attempts % 10000000 == 0) {
            clock_t current_time = clock();
            double elapsed = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            double hash_rate = total_attempts / elapsed;
            printf("Progress: %llu attempts, %.2f MH/s\n", total_attempts, hash_rate / 1000000.0);
        }
    }
    
    // No solution found
    printf("EXHAUSTED %llu\n", total_attempts);
    
cleanup:
    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result);
    cudaFree(d_found_flag);
    
    return 0;
}
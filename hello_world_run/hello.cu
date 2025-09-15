#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", tid);
}

int main() {
    const int num_threads = 256;
    const int threads_per_block = 256;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    cudaError_t err;

    hello_kernel<<<num_blocks, threads_per_block>>>();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 1000
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5

__global__ void mandelbrot_kernel(int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * WIDTH + idx;

    if (idx >= WIDTH || idy >= HEIGHT) return;

    double x0 = X_MIN + (X_MAX - X_MIN) * idx / (double)WIDTH;
    double y0 = Y_MIN + (Y_MAX - Y_MIN) * idy / (double)HEIGHT;
    double x = 0.0, y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < MAX_ITER) {
        double xtemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        iter++;
    }

    output[index] = iter;
}

void write_ppm(const char *filename, int *data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        unsigned char color = (data[i] == MAX_ITER) ? 0 : (data[i] % 255);
        fputc(color, fp); // R
        fputc(color, fp); // G
        fputc(color, fp); // B
    }
    fclose(fp);
}

int main() {
    int *d_output, *h_output;
    size_t size = WIDTH * HEIGHT * sizeof(int);

    // Allocate host and device memory
    h_output = (int*)malloc(size);
    cudaMalloc(&d_output, size);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    mandelbrot_kernel<<<gridDim, blockDim>>>(d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Write to PPM file
    write_ppm("mandelbrot.ppm", h_output, WIDTH, HEIGHT);

    // Clean up
    free(h_output);
    cudaFree(d_output);

    printf("Mandelbrot set rendered to mandelbrot.ppm\n");
    return 0;
}
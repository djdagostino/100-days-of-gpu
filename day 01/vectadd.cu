'''
Summary:
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.

Learned:

Basics of writing a CUDA kernel.
Understanding of grid, block, and thread hierarchy in CUDA.
How to allocate and manage device (GPU) memory using cudaMalloc, cudaMemcpy, and cudaFree.
'''

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // Size of vectors (e.g., 1M elements)
    const size_t bytes = N * sizeof(float);

    // Host memory allocation
    std::vector<float> h_A(N, 1.0f); // Initialize to 1.0
    std::vector<float> h_B(N, 2.0f); // Initialize to 2.0
    std::vector<float> h_C(N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            break;
        }
    }

    std::cout << (success ? "Test PASSED\n" : "Test FAILED\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

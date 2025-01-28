'''
Summary:
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.

Learned:

Basics of writing a CUDA kernel.
Understanding of grid, block, and thread hierarchy in CUDA.
How to allocate and manage device (GPU) memory using cudaMalloc, cudaMemcpy, and cudaFree.
'''

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    
}
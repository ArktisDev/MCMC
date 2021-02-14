#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <string>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "distributions.cu"

// Workaround for intellisense and linter.
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();  // workaround __syncthreads warning
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#else
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif

// from some other tutorial I found, it is pretty handy.
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(const unsigned long long seed, curandState *rand_state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(seed, id, 0, rand_state + id);
}

__global__ void MHSequence(float *s, const int N, curandState *rand_state, const float sigma_step) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = *(rand_state + id);

    float p = curand_uniform(&local_state);
    float prob = WS_pdf(p);
    *(s + 0 + id * N) = p;

    for (size_t i = 1; i < N; i++) {
        float dp = sigma_step * curand_normal(&local_state);
        float nprob = WS_pdf(p + dp);

        if (curand_uniform(&local_state) <= nprob / prob) {
            p += dp;
            prob = nprob;
        }

        *(s + i + id * N) = p;
    }
}

// generates a sequence of random numbers using built in gaussian distribution
__global__ void trueGaussian(float *s, const int N, curandState *rand_state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = *(rand_state + id);

    for (size_t i = 0; i < N; i++) {
        *(s + i + id * N) = curand_normal(&local_state);
    }
}

int main() {
     // Force CUDA context establishment. Not necessary but nvprof output is more understandable.
    checkCudaErrors(cudaFree(0));

    const int blocks = 1;
    const int blocksize = 1;
    const int total_threads = blocks * blocksize;
    
    // Initialize random state
    curandState *d_rand_state; // d_ prefix is a device pointer
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, total_threads * sizeof(curandState)));
    const unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rand_init KERNEL_ARG2(blocks, blocksize) (seed, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    // sequence length per thread
    const long long N = 1<<20;

    float *s;
    float *d_s;
    float sigma_step = 2.5f;

    //treat 1D array like 2D array because it is just easier
    //arr[i][j] becomes arr[j * cols + i]
    s = (float *) malloc(N * total_threads * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&d_s, N * total_threads * sizeof(float)));
    
    MHSequence KERNEL_ARG2(blocks, blocksize) (d_s, N, d_rand_state, sigma_step);
    //trueGaussian KERNEL_ARG2(blocks, blocksize) (d_s, N, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(s, d_s, N * total_threads * sizeof(float), cudaMemcpyDeviceToHost));

    std::string datafile("distribution.data");
    std::ofstream data(datafile);

    for (size_t i = 0; i < total_threads; i++) {
        for (size_t j = 0; j < N; j++) {
            data << *(s + j + i * N) << "\n";
        }
    }

    data << std::flush;
    data.close();
    
    // free memory
    free(s);
    checkCudaErrors(cudaFree(d_s));

    return 0;
}
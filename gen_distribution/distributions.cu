#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// mean = 0.0f, variance = 1.0f, not normalised because it doesn't have to be
__device__ float normal_pdf(float x) {
    return exp(x * x / -2.0f);
}

__device__ float WS_pdf(float x) {
    if (x >= 0) {
        return 1 / (1 + exp((x - 2.0f) / 0.54f));
    } else {
        return 0;
    }
}
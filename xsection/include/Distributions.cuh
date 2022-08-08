#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include "Common.cuh"

// Woods Saxon Distribution
__device__ __host__ float WoodsSaxon(float r, float c, float a) {
    if (r < 0.0f) return 0.0f;
    return 1 / (1 + expf( (r - c) / a ));
}

// r^2 * Woods Saxon Distribution
__device__ __host__ float R2WoodsSaxon(float r, float c, float a) {
    if (r < 0.0f) return 0.0f;
    return r * r / (1 + expf( (r - c) / a ));
}

// Gaussian distribution
__device__ __host__ float Gaussian(float r, float variance) {
    if(r < 0.0f) return 0.0f;
    return expf(- r * r / (2 * variance));
}

// r^2 * Gaussian distribution
__device__ __host__ float R2Gaussian(float r, float variance) {
    if(r < 0.0f) return 0.0f;
    return r * r * expf(- r * r / (2 * variance));
}

// __device__ __host__ float ProjectedGaussian(float r) {
//     if (r < 0.0f) return 0.0f;
//     return r * expf(- r * r / (2 * 1 * 1));
// }

#endif // DISTRIBUTIONS_H

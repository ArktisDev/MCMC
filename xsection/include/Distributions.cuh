#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include "Common.cuh"

// Woods Saxon Distribution
__device__ __host__ float WoodsSaxon(float r, float a, float c) {
    if (r < 0.0f) return 0.0f;
    return 1 / (1 + expf( (r - c) / a ));
}

// r^2 * Woods Saxon Distribution
__device__ __host__ float R2WoodsSaxon(float r, float a, float c) {
    if (r < 0.0f) return 0.0f;
    return r * r / (1 + expf( (r - c) / a ));
}

// PDF used for integration
__device__ __host__ float pdf(float r) {
    return R2WoodsSaxon(r, 0.54f, 2.0f);
}

#endif // DISTRIBUTIONS_H
#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "common.cuh"
#include "metropolis.cuh"

// Woods Saxon Distribution
__device__ __host__ float WoodsSaxon(float r, float a, float c) {
    if (r < 0.0f) return 0.0f;
    return 1 / (1 + exp( (r - c) / a ));
}

// r^2 * Woods Saxon Distribution
__device__ __host__ float R2WoodsSaxon(float r, float a, float c) {
    if (r < 0.0f) return 0.0f;
    return r * r / (1 + exp( (r - c) / a ));
}

// PDF used for integration
__device__ __host__ float pdf(float r) {
    return R2WoodsSaxon(r, 0.54f, 2.0f);
}

__global__ void MCIntegrate(float* prevSample, float* invPrevPDF, curandStateXORWOW* randState, float* resultBuffer, int samples) {
   int threadId = threadIdx.x + blockIdx.x * blockDim.x;
   
   curandStateXORWOW localRandState = randState[threadId];
   float localPrevSample = prevSample[threadId];
   float localInvPrevPDF = invPrevPDF[threadId];
   
   float sum = 0.0f;
   float err = 0.0f;
   
   // use Kahan Summation
   for (int i = 0; i < samples; i++) {
      // do an M-H iteration
      MetropolisHastingsStep<pdf>(&localPrevSample, &localInvPrevPDF, 2.32 * 2.32 + 0.01, &localRandState);
      
      float y = localPrevSample * localPrevSample * localPrevSample - err;
      float t = sum + y;
      err = (t - sum) - y;
      sum = t;
   }
   
   resultBuffer[threadId] = (sum - err) / samples;
   
   randState[threadId] = localRandState;
   prevSample[threadId] = localPrevSample;
   invPrevPDF[threadId] = localInvPrevPDF;
}

#endif // INTEGRATE_H

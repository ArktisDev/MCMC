
#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "common.cuh"

// Initialize the random state of the curandStateXORWOW variable using a seed
__global__ void RandStateInit(curandStateXORWOW *randState, unsigned long long seed) {
   int threadId = threadIdx.x + blockIdx.x * blockDim.x;
   
   // Each thread gets the same seed, a different sequence number, no offset
   // By documentation (https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-1)
   // this should give a random sequence of *bits* with a period 2 ^ 190
   // Each call to a distribution sample will use some number of those bits.
   // From what I can tell, uniform and normal XORWOW each use 32 bits from the sequence,
   // so the true period for random numbers is actually 2 ^ 185
   // Sequence number (threadId) will bring the state to position 2^67 * sequence in the overall sequence of bits.
   // So each thread actually sits at 2^62 * sequence in the sequence of actual generated numbers.
   // 2 ^ 62 ~= 4 * 10^18, so really there will be no repeat random numbers
   curand_init(seed, threadId, 0, &(randState[threadId]));
}

// Initialize the state of the metropolis algorithm using initSample
template<float (* pdf)(float)>
__global__ void MetropolisInit(float* prevSample, float* invPrevPDF, int initSample) {
   int threadId = threadIdx.x + blockIdx.x * blockDim.x;
   
   prevSample[threadId] = initSample;
   invPrevPDF[threadId] = 1.0f / pdf(initSample);
}

// Given the previous sample and saved previous sample PDF, use MH algorithm and
// random state to generate next sample in MH chain.
template<float (* pdf)(float)>
__device__ void MetropolisHastingsStep(float* prevSample, float* invPrevPDF, float stepSize, curandStateXORWOW* randState) {
   // generate next potential sample
   float nextSample = *prevSample + stepSize * curand_normal(randState);
   float nextPDF = pdf(nextSample);
   
   // acceptance condition for the new sample
   if (nextPDF * (*invPrevPDF) >= curand_uniform(randState)) {
      *prevSample = nextSample;
      *invPrevPDF = 1 / nextPDF;
   }
}

// Using previous definitions, run the MH chain for some number of iterations to warm up and reach "steady state"
template<float (* pdf)(float)>
__global__ void WarmupMetropolis(float * prevSample, float* invPrevPDF, float stepsize, curandStateXORWOW* randState, int iterations) {
   int threadId = threadIdx.x + blockIdx.x * blockDim.x;
   
   // make local copies of important variables to eliminate global memory usage
   curandStateXORWOW localRandState = randState[threadId];
   float localPrevSample = prevSample[threadId];
   float localInvPrevPDF = invPrevPDF[threadId];
   
   for (int i = 0; i < iterations; i++) {
      // do an M-H iteration
      MetropolisHastingsStep<pdf>(&localPrevSample, &localInvPrevPDF, 1, &localRandState);
   }
   
   // store back results
   randState[threadId] = localRandState;
   prevSample[threadId] = localPrevSample;
   invPrevPDF[threadId] = localInvPrevPDF;
}

#endif // METROPOLIS_H
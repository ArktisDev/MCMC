#ifndef INTEGRATER2_H
#define INTEGRATER2_H

#include "Common.cuh"
#include "Metropolis.cuh"

template <int nNucleonsA, int nNucleonsB, int totalThreads, PDF... pdfs>
__global__ void MCIntegrate_R2(float* __restrict__ prevSample, float stepsize, curandStateXORWOW* __restrict__ randState, float* __restrict__ resultBuffer, int samples)
{
    const int nNucleons = nNucleonsA + nNucleonsB;

    // Static assert here that length of pdfs matches nNucleons
    static_assert(nNucleons == sizeof...(pdfs), "Incorrect number of pdf's to MCIntegrate_R2");
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandStateXORWOW localRandState = randState[threadId];
    float localPrevSample[nNucleons];
    float localInvPrevPDF[nNucleons];
    
    CopyLocalMHVars<totalThreads, 0, pdfs...>(threadId, prevSample, localPrevSample, localInvPrevPDF);
    
    float sum[nNucleons];
    float err[nNucleons];

    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++) {
        sum[n] = 0.0f;
        err[n] = 0.0f;
    }
    
    // use Kahan Summation to reduce (mostly eliminate) error
    for (int s = 0; s < samples; s++)
    {
        GibbsStep<0, pdfs...>(localPrevSample, localInvPrevPDF, stepsize, &localRandState);
        
        #pragma unroll nNucleons
        for (int n = 0; n < nNucleons; n++) {
            float y = localPrevSample[n] * localPrevSample[n];

            y -= err[n];
            
            float t = sum[n] + y;
            err[n] = (t - sum[n]) - y;
            sum[n] = t;
        }
    }

    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++) {
        resultBuffer[totalThreads * n + threadId] = (sum[n] - err[n]) / samples;
    }

    // store back results
    randState[threadId] = localRandState;
    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++)
    {
        prevSample[totalThreads * n + threadId] = localPrevSample[n];
    }
}

#endif // INTEGRATER2_H

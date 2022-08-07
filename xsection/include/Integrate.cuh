#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "Common.cuh"
#include "Metropolis.cuh"

// r in fm^2
__device__ __host__ float GammaNNR2(float r) {
    const float beta = 0.2f; // in fm^2
    const float sigmaNN = 4.3f; // in fm^2 = 10 * mb
    return sigmaNN / (4 * PI * beta) * expf(- r / (2.f * beta));
}

__device__ __host__ float GammaNN(float dx, float dy) {
    return GammaNNR2(dx * dx + dy * dy);
}

template <int nNucleonsA, int nNucleonsB, int totalThreads, PDF... pdfs>
__global__ void MCIntegrate_S_AB(float* __restrict__ prevSample, float stepsize, curandStateXORWOW* __restrict__ randState, float* __restrict__ resultBuffer, int samples, float impactParameter)
{
    const int nNucleons = nNucleonsA + nNucleonsB;

    // Static assert here that length of pdfs matches nNucleons
    static_assert(nNucleons == sizeof...(pdfs), "Incorrect number of pdf's to MCIntegrate_S_AB");
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandStateXORWOW localRandState = randState[threadId];
    float localPrevSample[nNucleons];
    float localInvPrevPDF[nNucleons];
    
    CopyLocalMHVars<totalThreads, 0, pdfs...>(threadId, prevSample, localPrevSample, localInvPrevPDF);
    
    float sum = 0.0f;
    float err = 0.0f;
    
    // use Kahan Summation to reduce (mostly eliminate) error
    for (int s = 0; s < samples; s++)
    {
        GibbsStep<0, pdfs...>(localPrevSample, localInvPrevPDF, stepsize, &localRandState);
        
        
        // TODO: Store only values for nNucleonsB. nNucleonsA is independent and can be generated on the fly. This halves number of registers needed
        // For choosing theta, phi of nucleon
        // https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
        float u0[nNucleons];
        float u1[nNucleons];
        #pragma unroll nNucleons
        for (int n = 0; n < nNucleons; n++) {
            u0[n] = curand_uniform(&localRandState);
            u1[n] = curand_uniform(&localRandState);
            
            // no need to keep u0 to calculate z-coordinate since z-coordinate will be discarded anyways
            // now u0[n] is what to multiply radial coordinates by because of amount of position "in z direction"
            u0[n] = 1 - 2 * u0[n];
            u0[n] = sqrtf(1 - u0[n] * u0[n]);
            
            // premultiply u1 by 2 for use in sinpif and cospif functions
            u1[n] = 2 * u1[n]; 
        }
        
        float y = 1;
        
        #pragma unroll nNucleonsA
        for (int i = 0; i < nNucleonsA; i++) {
            
            #pragma unroll nNucleonsB
            for (int j = 0; j < nNucleonsB; j++) {
                float ux = localPrevSample[i] * u0[i] * cospif(u1[i]);
                float uy = localPrevSample[i] * u0[i] * sinpif(u1[i]);
                float sx = localPrevSample[j + nNucleonsA] * u0[j + nNucleonsA] * cospif(u1[j + nNucleonsA]);
                float sy = localPrevSample[j + nNucleonsA] * u0[j + nNucleonsA] * sinpif(u1[j + nNucleonsA]);
                // Because of symmetry, it is fine to assume `impactParameter` to always be towards right
                y *= (1 - GammaNN(impactParameter + ux - sx, uy - sy));
            }
            
        }
        
        // This is part of Kahan
        y -= err;
        
        float t = sum + y;
        err = (t - sum) - y;
        sum = t;
    }
    
    resultBuffer[threadId] = (sum - err) / samples;

    // store back results
    randState[threadId] = localRandState;
    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++)
    {
        prevSample[totalThreads * n + threadId] = localPrevSample[n];
    }
}

template <int nNucleonsA, int nNucleonsB, int totalThreads, PDF... pdfs>
__global__ void MCIntegrate_R2(float* __restrict__ prevSample, float stepsize, curandStateXORWOW* __restrict__ randState, float* __restrict__ resultBuffer, int samples)
{
    const int nNucleons = nNucleonsA + nNucleonsB;

    // Static assert here that length of pdfs matches nNucleons
    static_assert(nNucleons == sizeof...(pdfs), "Incorrect number of pdf's to MCIntegrate_S_AB");
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandStateXORWOW localRandState = randState[threadId];
    float localPrevSample[nNucleons];
    float localInvPrevPDF[nNucleons];
    
    CopyLocalMHVars<totalThreads, 0, pdfs...>(threadId, prevSample, localPrevSample, localInvPrevPDF);
    
    float sum[nNucleons];
    float err[nNucleons];

    #pragma unroll nNucleons
    for (int i = 0; i < nNucleons; i++) {
        sum[i] = 0.0f;
        err[i] = 0.0f;
    }
    
    // use Kahan Summation to reduce (mostly eliminate) error
    for (int s = 0; s < samples; s++)
    {
        GibbsStep<0, pdfs...>(localPrevSample, localInvPrevPDF, stepsize, &localRandState);
        
        #pragma unroll nNucleons
        for (int n = 0; n < nNucleons; n++) {
            // This is part of Kahan

            float y = localPrevSample[n] * localPrevSample[n];

            y -= err[n];
            
            float t = sum[n] + y;
            err[n] = (t - sum[n]) - y;
            sum[n] = t;
        }
    }

    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++) {
        resultBuffer[totalThreads * n + threadID] = (sum[n] - err[n]) / samples;

    }

    // store back results
    randState[threadId] = localRandState;
    #pragma unroll nNucleons
    for (int n = 0; n < nNucleons; n++)
    {
        prevSample[totalThreads * n + threadId] = localPrevSample[n];
    }
}

#endif // INTEGRATE_H

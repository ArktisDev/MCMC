#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "Common.cuh"
#include "Metropolis.cuh"

#include <tuple>

// r in fm^2
__device__ __host__ float GammaNNR2(float r) {
    const float beta = 0.2f; // in fm^2
    const float sigmaNN = 4.3f; // in fm^2 = 10 * mbarns
    return sigmaNN / (4 * PI * beta) * expf(- r / (2.f * beta));
}

__device__ __host__ float GammaNN(float r) {
    return GammaNNR2(r * r);
}


__device__ __host__ float GammaNN(float dx, float dy) {
    return GammaNNR2(dx * dx + dy * dy);
}

template <int nNucleonsA, int nNucleonsB, int totalThreads, PDF... pdfs>
__global__ void MCIntegrate_S_AB(float *prevSample, float stepsize, curandStateXORWOW *randState, float *resultBuffer, int samples, float impactParameter)
{
    const int nNucleons = nNucleonsA + nNucleonsB;
    if constexpr (sizeof...(pdfs) == 1)
    {
        int threadId = threadIdx.x + blockIdx.x * blockDim.x;

        curandStateXORWOW localRandState = randState[threadId];
        float localPrevSample[nNucleons];
		float localInvPrevPDF[nNucleons];
        
        // Just pull first PDF from pack which only contains 1 pdf.
		// somewhat upsetting we can't just pdf = pdfs[0];
		constexpr PDF pdf = std::get<0>(std::forward_as_tuple(pdfs...));
        
        // copy data into local copies
		#pragma unroll nNucleons
		for (int n = 0; n < nNucleons; n++)
		{
			localPrevSample[n] = prevSample[totalThreads * n + threadId];
			localInvPrevPDF[n] = 1 / pdf(localPrevSample[n]);
		}
        
        float sum = 0.0f;
        float err = 0.0f;
        
        // use Kahan Summation
        for (int i = 0; i < samples; i++)
        {
            // Use gibbs sampling to update all variables
			// In our case since P(a, b, ...) = P(a) * P(b) * ...
			// each dimension can just be updated alone
            #pragma unroll nNucleons
			for (int n = 0; n < nNucleons; n++)
			{
				MetropolisHastingsStep<pdf>(&(localPrevSample[n]), &(localInvPrevPDF[n]), stepsize, &localRandState);
			}

            float y = (localPrevSample[0] - localPrevSample[1] * localPrevSample[1]) - err;
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
    } else {
        // Multiple PDF's given
		// static assert here that length of pdf matches nNucleons
		static_assert(nNucleons == sizeof...(pdfs), "Incorrect number of pdf's");
        
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
            
            // For choosing theta, phi of nucleon
            // https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
            float u0[nNucleons];
            float u1[nNucleons];
            #pragma unroll nNucleons
            for (int n = 0; n < nNucleons; n++) {
                u0[n] = curand_uniform(&localRandState);
                u1[n] = curand_uniform(&localRandState);
                
                // no need to keep u0 since z-coordinate will be discarded anyways
                // now u0[n] is what to multiply radial coordinates by because of amount of position "in z direction"
                u0[n] = 2 * u0[n] - 1;
                u0[n] = sqrt(1 - u0[n] * u0[n]);
                
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
}

#endif // INTEGRATE_H

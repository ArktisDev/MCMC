#include "common.cuh"
#include "statistics.cuh"
#include "timing.hpp"
#include "metropolis.cuh"
#include "integrate.cuh"

#include <iostream>

constexpr PDF pdflist[] = {pdf};

int main() {
    const int nNeutronA = 1;
    const int nProtonA = 0;
    const int nNeutronB = 0;
    const int nProtonB = 0;
    
    const int nNucleonA = nNeutronA + nProtonA;
    const int nNucleonB = nNeutronB + nProtonB;
    const int nNucleons = nNucleonA + nNucleonB;
    
    // For 1 simulation kernel
    const int totalBlocks = 1 << 12; // 4096
    const int threadsPerBlock = 1 << 7;    // 128, don't go below this per block (empirical)
    const int samplesPerThread = 1 << 10; // don't go below 1<<9 really (empirical)
    
    const int totalRuns = 1 << 6;  // number of times to iterate the simulation kernel
    
    const int samplesForMix = 1 << 14; // number of times to iterate MH kernel for mixing
    
    // seed for CUDA RNG
    const uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    //const uint64_t seed = 8 * 2474033889142906;
    
    
    
    
    
    
    
    
    // If you know the actual result for comparison
    //const float actualResult = 6.424247445544025f;
    //const float actualResult = 2.3317236182528682f;
    const float actualResult = 20.35336559958019f;
    
    const int64_t totalThreads = totalBlocks * threadsPerBlock;
    const int64_t samplesPerRun = samplesPerThread * totalThreads;
    const int64_t totalSamples = samplesPerRun * totalRuns;
    
    std::cout << "Total threads: " << totalThreads << std::endl;
    std::cout << "Samples per thread: " << samplesPerThread << std::endl;
    std::cout << "Total runs: " << totalRuns << std::endl;
    std::cout << "Samples per run: " << samplesPerRun << std::endl;
    std::cout << "Total samples: " << totalSamples << std::endl;
    std::cout << "Total samples log10():" << std::log10(totalSamples) << std::endl;
    
    dim3 blocks(totalBlocks);
    dim3 threads(threadsPerBlock);




    
    // Start of initialization
    
    // A very basic class to do timing
    Timer timer;
    
    // Keep track of random state and data for metropolis hastings
    // Rather than recalculate 1 / pdf(r_prev), just store it between kernel invocations
    curandStateXORWOW *d_randState;
    float *d_prevSample;
    
    cudaMalloc((void **)&d_randState, totalThreads * sizeof(curandStateXORWOW));
    cudaCheckError();
    
    cudaMalloc((void **)&d_prevSample, nNucleons * totalThreads * sizeof(float));
    cudaCheckError();
    
    // Init arrays
    RandStateInit<<<blocks, threads>>>(d_randState, seed);
    cudaCheckError();
    InitSampleArray<<<blocks, threads>>>(d_prevSample, 1.0f, nNucleons, totalThreads);
    cudaCheckError();
    
    cudaDeviceSynchronize();
    cudaCheckError();
    
    float *d_resultBuffer1;
    float *d_resultBuffer2;
    float *h_resultBuffer1;
    float *h_resultBuffer2;
    
    cudaMalloc((void **)&d_resultBuffer1, totalThreads * sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_resultBuffer2, totalThreads * sizeof(float));
    cudaCheckError();
    
    h_resultBuffer1 = (float *) malloc(totalThreads * sizeof(float));
    h_resultBuffer2 = (float *) malloc(totalThreads * sizeof(float));
    
    float* d_resultBuffers[2] = {d_resultBuffer1, d_resultBuffer2};
    float* h_resultBuffers[2] = {h_resultBuffer1, h_resultBuffer2};
    
    cudaStream_t computeStream, dataStream;
    cudaStreamCreate(&computeStream);
    cudaCheckError();
    cudaStreamCreate(&dataStream);
    cudaCheckError();
    
    cudaEvent_t batches[2];
    for (cudaEvent_t& event : batches) {
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        cudaCheckError();
    }
    
    float runAverages[totalRuns];
    
    // Warmup markov chain
    
    timer.start();
    
    // Start running the chain before taking samples from it, this "mixes" the chain
    // This is an important step, and if we were doing this a little bit better
    // we'd even randomize the initial states of the chain
    // This can be done by treating it as a discrete pdf, and sampling according to that (lars can do that btw lol)
    // Probably it isn't worth it though since we can just mix the chain "enough"
    WarmupMetropolis<nNucleons, totalThreads, pdflist[0]><<<blocks, threads>>>(d_prevSample, 2.2f, d_randState, samplesForMix);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    
    std::cout << "Warmup took " << timer.elapsedMilli() << "ms" << std::endl;
    
    // Setup done, now run the integration
    timer.start();
    
    // // Launch the first run
    // MCIntegrate<<<blocks, threads, 0, computeStream>>>(d_prevSample, d_invPrevPDF, d_randState, d_resultBuffer1, samplesPerThread);
    // cudaCheckError();
    // cudaEventRecord(batches[0], computeStream);
    // cudaCheckError();
    
    // for (int run = 1; run < totalRuns; run++) {
    //     // Launch a new run
    //     MCIntegrate<<<blocks, threads, 0, computeStream>>>(d_prevSample, d_invPrevPDF, d_randState, d_resultBuffers[run % 2], samplesPerThread);
    //     cudaCheckError();
    //     cudaEventRecord(batches[run % 2], computeStream);
    //     cudaCheckError();
        
    //     // Wait on previous kernel to finish
    //     cudaEventSynchronize(batches[(run + 1) % 2]);
    //     cudaCheckError();
        
    //     // Process data from that event
    //     cudaMemcpyAsync(h_resultBuffers[(run - 1) % 2], d_resultBuffers[(run - 1) % 2], totalThreads * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
    //     cudaCheckError();
    //     cudaStreamSynchronize(dataStream);
    //     cudaCheckError();
        
    //     runAverages[run - 1] = Average(h_resultBuffers[(run - 1) % 2], totalThreads);
        
    //     // when done processing, just let loop again and enqueue another kernel, or let exit because we've enqueued enough kernels
    // }
    
    // // when exiting the loop, there is still one running kernel, with id run = totalRuns - 1.
    // cudaEventSynchronize(batches[(totalRuns - 1) % 2]);
    // cudaCheckError();
    
    // // process the data from that event
    // cudaMemcpyAsync(h_resultBuffers[(totalRuns - 1) % 2], d_resultBuffers[(totalRuns - 1) % 2], totalThreads * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
    // cudaCheckError();
    // cudaStreamSynchronize(dataStream);
    // cudaCheckError();
    
    // runAverages[totalRuns - 1] = Average(h_resultBuffers[(totalRuns - 1) % 2], totalThreads);
    
    // int64_t elapsedTime = timer.elapsedMilli();
    
    
    
    
    // // Statistics
    
    
    // float finalAverage = Average(runAverages, totalRuns);
    // float finalVariance = Variance(runAverages, finalAverage, totalRuns);
    // std::cout << "Final average: " << finalAverage << std::endl;
    // std::cout << "Final variance: " << finalVariance << std::endl;
    // std::cout << "Final stdev: " << sqrt(finalVariance) << std::endl;
    // std::cout << "Final stderr: " << sqrt(finalVariance / totalRuns) << std::endl;
    // std::cout << "Actual Error: " << finalAverage - actualResult << std::endl;
    
    
    // std::cout << "Elapsed time: " << elapsedTime / 1000.0 << "s" << std::endl;
    // std::cout << "Samples/s: " << totalSamples / elapsedTime / 1e6 << " GS/s" << std::endl;
    
    
    
    // Cleanup
    
    cudaStreamDestroy(computeStream);
    cudaCheckError();
    cudaStreamDestroy(dataStream);
    cudaCheckError();
    
    for (cudaEvent_t& event : batches) {
        cudaEventDestroy(event);
        cudaCheckError();
    }
    
    cudaFree(d_randState);
    cudaCheckError();
    cudaFree(d_prevSample);
    cudaCheckError();
    cudaFree(d_resultBuffer1);
    cudaCheckError();
    cudaFree(d_resultBuffer2);
    cudaCheckError();
    
    free(h_resultBuffer1);
    free(h_resultBuffer2);
    
    return 0;
}
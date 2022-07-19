#include "Common.cuh"
#include "Statistics.cuh"
#include "Timing.cuh"
#include "ProgressBar.cuh"
#include "IntegrationOutputHandler.cuh"
#include "Metropolis.cuh"
#include "Integrate.cuh"
#include "Distributions.cuh"

#include <iostream>

// This define for expanding the pdflist in variadic template
#define pdflist pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf, pdf
// In the future it might be useful to include a dummy header for compiling
// where the content of the dummy header is just the define for pdflist.
// That way, an external script can modify the dummy header for the specific
// situation to be compiled, then run the compiler. This makes it so we don't
// have to mess around passing the defines via command line, which is somewhat
// difficult and easy to mess up. 

int main(int argc, char** argv) {
    const int nNeutronA = 6;
    const int nProtonA = 6;
    const int nNeutronB = 6;
    const int nProtonB = 6;
    
    // For 1 simulation kernel
    const int totalBlocks = 1 << 10; // 1024
    const int threadsPerBlock = 1 << 7;    // 128, don't go below this per block (empirical)
    const int samplesPerThread = 1 << 8; // don't go below 1<<9 really (empirical)
    
    const int totalRuns = 1 << 4;  // number of times to iterate the simulation kernel
    
    const int samplesForMix = 1 << 13; // number of times to iterate MH kernel for mixing
    
    // Range of impact parameters to sample
    const float b0 = 0;
    const float db = 0.25;
    // number of db's to step
    const int ndbs = 40;
    
    // Output file
    std::string outDir = "../data/";
    std::string outFile = "S_AB.dat";
    
    // If an arg is supplied, set that to be the output file
    if (argc == 2) {
        outFile = argv[1];
    }
    
    // seed for CUDA RNG
    const uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    //const uint64_t seed = 8 * 2474033889142906;
    
    
    
    
    
    
    const int nNucleonsA = nNeutronA + nProtonA;
    const int nNucleonsB = nNeutronB + nProtonB;
    const int nNucleons = nNucleonsA + nNucleonsB;
    
    const int64_t totalThreads = totalBlocks * threadsPerBlock;
    const int64_t samplesPerRun = samplesPerThread * totalThreads;
    const int64_t totalSamples = samplesPerRun * totalRuns;
    
    // TODO: maybe log stuff like this in metadata in some file
    std::cout << "Total threads: " << totalThreads << std::endl;
    std::cout << "Samples per thread: " << samplesPerThread << std::endl;
    std::cout << "Total runs: " << totalRuns << std::endl;
    std::cout << "Samples per run: " << samplesPerRun << std::endl;
    std::cout << "Total samples: " << totalSamples << std::endl;
    std::cout << "Total samples log10():" << std::log10(totalSamples) << std::endl;
    
    dim3 blocks(totalBlocks);
    dim3 threads(threadsPerBlock);




    
    // Start of initializing CUDA variables
    
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
    WarmupMetropolis<nNucleons, totalThreads, pdflist><<<blocks, threads>>>(d_prevSample, 2.2f, d_randState, samplesForMix);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    
    std::cout << "Warmup took " << timer.elapsedMilli() << "ms" << std::endl;
    
    
    std::vector<std::string> headers = {"ImpactParameter", "S_AB", "Stderr(S_AB)"};
    IntegrationOutputHandler handler(headers.size(), headers, outDir, outFile);
    
    ProgressBar progressBar(ndbs * totalRuns);
    
    std::cout << "Now starting integration" << std::endl;
    
    
    
    
    
    
    
    
    
    
    // Loop over wanted impact parameters
    // dbs = number of db's from b0 to current impact parameter
    for (int dbs = 0; dbs < ndbs; dbs++) {
        float b = b0 + dbs * db;
        // Setup done, now run the integration
        progressBar.IncrementProgress(1);
        progressBar.PrintBar();
        timer.start();
        
        // Launch the first run
        MCIntegrate_S_AB<nNucleonsA, nNucleonsB, totalThreads, pdflist><<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffer1, samplesPerThread, b);
        cudaCheckError();
        cudaEventRecord(batches[0], computeStream);
        cudaCheckError();
        
        for (int run = 1; run < totalRuns; run++) {
            progressBar.IncrementProgress(1);
            progressBar.PrintBar();
            // Launch a new run
            MCIntegrate_S_AB<nNucleonsA, nNucleonsB, totalThreads, pdflist><<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffers[run % 2], samplesPerThread, b);
            cudaCheckError();
            cudaEventRecord(batches[run % 2], computeStream);
            cudaCheckError();
            
            // Wait on previous kernel to finish
            cudaEventSynchronize(batches[(run + 1) % 2]);
            cudaCheckError();
            
            // Process data from that event
            cudaMemcpyAsync(h_resultBuffers[(run - 1) % 2], d_resultBuffers[(run - 1) % 2], totalThreads * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
            cudaCheckError();
            cudaStreamSynchronize(dataStream);
            cudaCheckError();
            
            runAverages[run - 1] = Average(h_resultBuffers[(run - 1) % 2], totalThreads);
            
            // when done processing, just let loop again and enqueue another kernel, or let exit because we've enqueued enough kernels
        }
        
        // when exiting the loop, there is still one running kernel, with id run = totalRuns - 1.
        cudaEventSynchronize(batches[(totalRuns - 1) % 2]);
        cudaCheckError();
        
        // process the data from that event
        cudaMemcpyAsync(h_resultBuffers[(totalRuns - 1) % 2], d_resultBuffers[(totalRuns - 1) % 2], totalThreads * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
        cudaCheckError();
        cudaStreamSynchronize(dataStream);
        cudaCheckError();
        
        runAverages[totalRuns - 1] = Average(h_resultBuffers[(totalRuns - 1) % 2], totalThreads);
        
        int64_t elapsedTime = timer.elapsedMilli();
        
        
        
        
        // Statistics
        
        
        float finalAverage = Average(runAverages, totalRuns);
        float finalVariance = Variance(runAverages, finalAverage, totalRuns);
        float finalStderr = sqrt(finalVariance / totalRuns);
        
        std::vector<float> results = {b, finalAverage, finalStderr};
        handler.AddRow(results);
        
        //std::cout << "Elapsed time: " << elapsedTime / 1000.0 << "s" << std::endl;
        //std::cout << "Integral Samples/s: " << totalSamples / elapsedTime / 1e6 << " GS/s" << std::endl;
    }
    
    progressBar.FinishBar();
    
    handler.WriteToFile();
    
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
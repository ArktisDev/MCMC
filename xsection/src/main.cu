#include "Common.cuh"
#include "Statistics.cuh"
#include "Timing.cuh"
#include "ProgressBar.cuh"
#include "IntegrationOutputHandler.cuh"
#include "Metropolis.cuh"
#include "Integrate.cuh"
#include "Distributions.cuh"
#include "TemplateBuilder.cuh"

#include <iostream>

// PDF used for integration, define other ones you want up here as actual functions and not lambdas
__device__ __host__ float pdf(float r) {
    //return R2WoodsSaxon(r, 0.54f, 1.535f);
    const float c = 2.f;
    const float a = 0.54f;
    return R2WoodsSaxon(r, c, a);
}

int main(int argc, char** argv) {
    const int nNeutronA = 2;
    const int nProtonA = 2;
    const int nNeutronB = 2;
    const int nProtonB = 2;
    
    const int nNucleonsA = nNeutronA + nProtonA;
    const int nNucleonsB = nNeutronB + nProtonB;
    const int nNucleons = nNucleonsA + nNucleonsB;
    
    constexpr static PDF nucleusAPdf = pdf;
    constexpr static PDF nucleusBPdf = pdf;
    constexpr static auto pdfArrA = TemplateBuilder::ConstArray<nNucleonsA>(nucleusAPdf);
    constexpr static auto pdfArrB = TemplateBuilder::ConstArray<nNucleonsB>(nucleusBPdf);
    constexpr static auto pdfArr  = TemplateBuilder::ConcatArray(pdfArrA, pdfArrB);
    
    // For 1 simulation kernel
    const int totalBlocks = 1 << 10; // 1024
    const int threadsPerBlock = 1 << 7;    // 128, don't go below this per block (empirical)
    const int samplesPerThread = 1 << 8; // don't go below 1<<9 really for MAX performance (empirical)
    
    const int totalRuns = 1 << 4;  // number of times to iterate the simulation kernel
    
    const int samplesForMix = 1 << 13; // number of times to iterate MH kernel for mixing
    
    // Range of impact parameters to sample
    const float b0 = 0;
    const float db = 0.1;
    // number of db's to step
    const int ndbs = 200;
    
    // seed for CUDA RNG
    const uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    //const uint64_t seed = 8 * 2474033889142906;
    
    // Output file
    std::string outDir = "../data/";
    std::string outFile = "S_AB.dat";
    
    // If an arg is supplied, set that to be the output file
    if (argc == 2) {
        outFile = argv[1];
    }

    const bool integrateRMS = true;
    const bool integrateSAB = false;

    // ===================================
    // || User variables end here       ||
    // ===================================

    const uint64_t totalThreads = totalBlocks * threadsPerBlock;
    const uint64_t samplesPerRun = samplesPerThread * totalThreads;
    const uint64_t totalSamples = samplesPerRun * totalRuns;
    const uint64_t totalSABSamples = totalSamples * ndbs;

    // TODO: maybe log stuff like this in metadata in some file
    std::cout << "=============================================" << std::endl;
    std::cout << "Per integral:" << std::endl;
    std::cout << "\tTotal threads: " << totalThreads << std::endl;
    std::cout << "\tSamples per thread: " << samplesPerThread << std::endl;
    std::cout << "\tTotal runs: " << totalRuns << std::endl;
    std::cout << "\tSamples per run: " << samplesPerRun << std::endl;
    std::cout << "\tTotal samples: " << totalSamples << std::endl;
    std::cout << "\tlog10(Total samples): " << std::log10(totalSamples) << std::endl;
    std::cout << "S_AB Integration:" << std::endl;
    std::cout << "\tTotal samples over all b's: " << totalSABSamples << std::endl;
    std::cout << "\tlog10(Total samples): " << std::log10(totalSABSamples) << std::endl;
    std::cout << "=============================================\n" << std::endl;
    
    constexpr static auto SABIntegrationFunction = TemplateBuilder::Make_SAB_Integration_Function<nNucleonsA, nNucleonsB, totalThreads, pdfArr>::type;
    constexpr static auto R2IntegrationFunction = TemplateBuilder::Make_R2_Integration_Function<nNucleonsA, nNucleonsB, totalThreads, pdfArr>::type;
    constexpr static auto WarmupMetropolisFunction = TemplateBuilder::Make_WarmupMetropolis_Function<nNucleons, totalThreads, pdfArr>::type;

    dim3 blocks(totalBlocks);
    dim3 threads(threadsPerBlock);
    
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

    // Warmup markov chain
    
    timer.Start();
    
    // Start running the chain before taking samples from it, this "mixes" the chain
    // This is an important step, and if we were doing this a little bit better
    // we'd even randomize the initial states of the chain
    // This can be done by treating it as a discrete pdf, and sampling according to that (lars can do that btw lol)
    // Probably it isn't worth it though since we can just mix the chain "enough"
    WarmupMetropolisFunction<<<blocks, threads>>>(d_prevSample, 2.2f, d_randState, samplesForMix);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    
    std::cout << "Metropolis chain warmup took " << timer.ElapsedMilli() << "ms\n" << std::endl;

    // RMS Integration
    if constexpr (integrateRMS) {
        float *d_resultBuffer1;
        float *d_resultBuffer2;
        float *h_resultBuffer1;
        float *h_resultBuffer2;
        
        cudaMalloc((void **)&d_resultBuffer1, totalThreads * nNucleons * sizeof(float));
        cudaCheckError();
        cudaMalloc((void **)&d_resultBuffer2, totalThreads * nNucleons * sizeof(float));
        cudaCheckError();
        
        h_resultBuffer1 = (float *) malloc(totalThreads * nNucleons * sizeof(float));
        h_resultBuffer2 = (float *) malloc(totalThreads * nNucleons * sizeof(float));
        
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

        float runAverages[nNucleons][totalRuns];

        std::cout << "Now starting RMS integration!" << std::endl;   

        ProgressBar progressBar(totalRuns);
        progressBar.IncrementProgress(1);
        progressBar.PrintBar();
        timer.Start();
            
        // Launch the first run
        R2IntegrationFunction<<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffer1, samplesPerThread);
        cudaCheckError();
        cudaEventRecord(batches[0], computeStream);
        cudaCheckError();
            
        for (int run = 1; run < totalRuns; run++) {
            progressBar.IncrementProgress(1);
            progressBar.PrintBar();
            // Launch a new run
            R2IntegrationFunction<<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffers[run % 2], samplesPerThread);
            cudaCheckError();
            cudaEventRecord(batches[run % 2], computeStream);
            cudaCheckError();
            
            // Wait on previous kernel to finish
            cudaEventSynchronize(batches[(run - 1) % 2]);
            cudaCheckError();
            
            // Process data from that event
            cudaMemcpyAsync(h_resultBuffers[(run - 1) % 2], d_resultBuffers[(run - 1) % 2], totalThreads * nNucleons * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
            cudaCheckError();
            cudaStreamSynchronize(dataStream);
            cudaCheckError();
            
            for (int n = 0; n < nNucleons; n++) {
                runAverages[n][run - 1] = Average((h_resultBuffers[(run - 1) % 2] + totalThreads * n), totalThreads);
            }
            
            // when done processing, just let loop again and enqueue another kernel, or let exit because we've enqueued enough kernels
        }
        
        // when exiting the loop, there is still one running kernel, with id run = totalRuns - 1.
        cudaEventSynchronize(batches[(totalRuns - 1) % 2]);
        cudaCheckError();
        
        // process the data from that event
        cudaMemcpyAsync(h_resultBuffers[(totalRuns - 1) % 2], d_resultBuffers[(totalRuns - 1) % 2], totalThreads * nNucleons * sizeof(float), cudaMemcpyDeviceToHost, dataStream);
        cudaCheckError();
        cudaStreamSynchronize(dataStream);
        cudaCheckError();
        
        for (int n = 0; n < nNucleons; n++) {
            runAverages[n][totalRuns - 1] = Average((h_resultBuffers[(totalRuns - 1) % 2] + totalThreads * n), totalThreads);
        }
        
        int64_t elapsedTime = timer.ElapsedMilli();

        progressBar.FinishBar();
        
        // Statistics

        float finalR2Average[nNucleons];
        float finalR2Stderr[nNucleons];
        float finalRMSAverage[nNucleons];
        float finalRMSStderr[nNucleons];

        for (int n = 0; n < nNucleons; n++) {
            finalR2Average[n] = Average(runAverages[n], totalRuns);
            finalR2Stderr[n] = std::sqrt(Variance(runAverages[n], finalR2Average[n], totalRuns) / totalRuns);
            finalRMSAverage[n] = std::sqrt(finalR2Average[n]);
            finalRMSStderr[n] = finalR2Stderr[n] / (2 * finalRMSAverage[n]);
        }

        std::cout << "RMS results" << std::endl;
        for (int n = 0; n < nNucleons; n++) {
            std::cout << "Nucleon " << n << " has RMS = " << finalRMSAverage[n] << " +/- " << finalRMSStderr[n] << std::endl;
        }

        // Cleanup
        
        cudaStreamDestroy(computeStream);
        cudaCheckError();
        cudaStreamDestroy(dataStream);
        cudaCheckError();
        
        for (cudaEvent_t& event : batches) {
            cudaEventDestroy(event);
            cudaCheckError();
        }
        
        cudaFree(d_resultBuffer1);
        cudaCheckError();
        cudaFree(d_resultBuffer2);
        cudaCheckError();
        
        free(h_resultBuffer1);
        free(h_resultBuffer2);
    } // End RMS Integration

    // S_AB Integration
    if constexpr (integrateSAB) {
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
        
        std::vector<std::string> headers = {"ImpactParameter", "S_AB", "Stderr(S_AB)"};
        IntegrationOutputHandler handler(headers.size(), headers, outDir, outFile);
        
        ProgressBar progressBar(ndbs * totalRuns);
        
        std::cout << "Now starting S_AB integration" << std::endl;    
        
        // Loop over wanted impact parameters
        // dbs = number of db's from b0 to current impact parameter
        for (int dbs = 0; dbs < ndbs; dbs++) {
            float b = b0 + dbs * db;
            // Setup done, now run the integration
            progressBar.IncrementProgress(1);
            progressBar.PrintBar();
            timer.Start();
            
            // Launch the first run
            SABIntegrationFunction<<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffer1, samplesPerThread, b);
            cudaCheckError();
            cudaEventRecord(batches[0], computeStream);
            cudaCheckError();
            
            for (int run = 1; run < totalRuns; run++) {
                progressBar.IncrementProgress(1);
                progressBar.PrintBar();
                // Launch a new run
                SABIntegrationFunction<<<blocks, threads, 0, computeStream>>>(d_prevSample, 2.2, d_randState, d_resultBuffers[run % 2], samplesPerThread, b);
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
            
            int64_t elapsedTime = timer.ElapsedMilli();
            
            
            
            
            // Statistics
            
            
            float finalAverage = Average(runAverages, totalRuns);
            float finalVariance = Variance(runAverages, finalAverage, totalRuns);
            float finalStderr = sqrt(finalVariance / totalRuns);
            
            std::vector<float> results = {b, finalAverage, finalStderr};
            handler.AddRow(results);
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
        
        cudaFree(d_resultBuffer1);
        cudaCheckError();
        cudaFree(d_resultBuffer2);
        cudaCheckError();
        
        free(h_resultBuffer1);
        free(h_resultBuffer2);
    } // End S_AB Integration
    
    // Cleanup metropolis variables
    cudaFree(d_randState);
    cudaCheckError();
    cudaFree(d_prevSample);
    cudaCheckError();
    return 0;
}
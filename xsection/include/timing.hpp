#ifndef TIMING_H
#define TIMING_H

#include "common.cuh"

#include <chrono>

// A class to make timing easier to read and do
class Timer {
    public:
        std::chrono::steady_clock::time_point t0;
        
        Timer() {};
        
        void start() {
            t0 = std::chrono::steady_clock::now();
        }
        
        int64_t elapsedMilli() {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        }
        
        int64_t elapsedMicro() {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
};

#endif // TIMING_H

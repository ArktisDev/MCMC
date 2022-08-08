#ifndef TIMING_H
#define TIMING_H

#include "Common.cuh"

#include <chrono>

// A class to make timing easier to read and do
class Timer {
    public:
        std::chrono::steady_clock::time_point t0;
        std::chrono::steady_clock::time_point t1;
        
        Timer() {};
        
        void Start() {
            t0 = std::chrono::steady_clock::now();
            t1 = t0;
        }

        void Stop() {
            t1 = std::chrono::steady_clock::now();
        }
        
        int64_t ElapsedMilli() {\
            // If timer hasn't been stopped, stop it
            if (t1 == t0) t1 = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        }
        
        int64_t ElapsedMicro() {
            // If timer hasn't been stopped, stop it
            if (t1 == t0) t1 = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
};

#endif // TIMING_H

#ifndef STATISTICS_H
#define STATISTICS_H

#include "common.cuh"

// Use Kahan summation to take sum of all the numbers in array `samples`
float Sum(float *samples, int n) {
    float res = 0.0f;
    float err = 0.0f;
    
    // Kahan summation algorithm
    for (int i = 0; i < n; i++) {
        float y = samples[i] - err;
        float t = res + y;
        err = (t - res) - y;
        res = t;
    }
    
    return res - err;
}

// Use Kahan summation to take average of all the numbers in array `samples`
float Average(float *samples, int n) {
    return Sum(samples, n) / n;
}

// Use Kahan summation to calculate variance of all the numbers in array `samples`
float Variance(float *samples, float avg, int n) {
    float res = 0.0f;
    float err = 0.0f;
    
    // Kahan summation algorithm
    for (int i = 0; i < n; i++) {
        float y = samples[i] - avg;
        y = y * y - err;
        float t = res + y;
        err = (t - res) - y;
        res = t;
    }
    
    return res / n;
}

// Use Kahan summation to calculate variance of all the numbers in array `samples`
float Variance(float *samples, int n) {
    float avg = Average(samples, n);
    float res = 0.0f;
    float err = 0.0f;
    
    // Kahan summation algorithm
    for (int i = 0; i < n; i++) {
        float y = samples[i] - avg;
        y = y * y - err;
        float t = res + y;
        err = (t - res) - y;
        res = t;
    }
    
    return res / n;
}

// Use Kahan summation to calculate autocorrelation of the all the numbers in array `samples`
float AutoCorrelation(float *samples, float avg, int lag, int n) {
    float num = 0.0f;
    float den = 0.0f;
    float err = 0.0f;
    
    // Kahan summation algorithm
    for (int i = 0; i < n - lag; i++) {
        float y = (samples[i] - avg) * (samples[i + lag] - avg) - err;
        float t = num + y;
        err = (t - num) - y;
        num = t;
    }
    
    err = 0.0f;
    
    // Kahan summation algorithm
    for(int i = 0; i < n; i++) {
        float y = (samples[i] - avg) * (samples[i] - avg) - err;
        float t = den + y;
        err = (t - den) - y;
        den = t;
    }
    
    return num / den;
}

// Use Kahan summation to calculate autocorrelation of the all the numbers in array `samples`
float AutoCorrelation(float *samples, int lag, int n) {
    float avg = Average(samples, n);
    float num = 0.0f;
    float den = 0.0f;
    float err = 0.0f;
    
    // Kahan summation algorithm
    for (int i = 0; i < n - lag; i++) {
        float y = (samples[i] - avg) * (samples[i + lag] - avg) - err;
        float t = num + y;
        err = (t - num) - y;
        num = t;
    }
    
    err = 0.0f;
    
    // Kahan summation algorithm
    for(int i = 0; i < n; i++) {
        float y = (samples[i] - avg) * (samples[i] - avg) - err;
        float t = den + y;
        err = (t - den) - y;
        den = t;
    }
    
    return num / den;
}

#endif // STATISTICS_H

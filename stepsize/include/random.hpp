#ifndef RANDOM_H
#define RANDOM_H

#include <thread>
#include <random>

double RandomUniform(double min, double max) {
    static thread_local std::mt19937 gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count() + static_cast<long long>(std::hash<std::thread::id>()(std::this_thread::get_id())));
    
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

double RandomNormal(double mean, double stdev) {
    static thread_local std::mt19937 gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count() + static_cast<long long>(std::hash<std::thread::id>()(std::this_thread::get_id())));
    
    std::normal_distribution<double> dist(mean, stdev);
    return dist(gen);
}

#endif // RANDOM_H

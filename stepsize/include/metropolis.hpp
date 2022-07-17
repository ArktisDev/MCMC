#ifndef METROPOLIS_H
#define METROPOLIS_H

#include <cmath>
#include <vector>

#include "random.hpp"

double PDF(double r) {
    double c = 2.0;
    double a = 0.54;
    
    if (r < 0.0) return 0.0;
    return r * r / (1 + std::exp( (r - c) / a ));
}

std::pair<double, std::vector<double>> MetropolisHastingsAlgorithm(int iterations, double stepSize) {
    double r = 1;
    
    int accepted = 0;
    
    std::vector<double> nums(iterations);
    
    for (int i = 0; i < iterations; i++) {
        double proposed = r + stepSize * RandomNormal(0, 1);
        
        double curPdf = PDF(r);
        double proposedPdf = PDF(proposed);
        
        if (proposedPdf > curPdf || RandomUniform(0, 1) < proposedPdf / curPdf) {
            r = proposed;
            accepted += 1;
        }
        
        nums[i] = r;
    }
    
    return std::pair<double, std::vector<double>>(1.0 * accepted / iterations, nums);
}

#endif // METROPOLIS_H
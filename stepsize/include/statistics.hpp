#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

double Average(std::vector<double> nums) {
    double sum = 0;
    int n = nums.size();
    
    for (int i = 0; i < n; i++) {
        sum += nums[i];
    }
    
    return sum / n;
}

double Variance(std::vector<double> nums, double average) {
    double sum = 0;
    int n = nums.size();
    
    for (int i = 0; i < n; i++) {
        double temp = (nums[i] - average);
        
        sum += temp * temp;
    }
    
    return sum / (n - 1);
}

double Variance(std::vector<double> nums) {
    double avg = Average(nums);
    
    return Variance(nums, avg);
}

double AutoCorrelation(std::vector<double> nums, int lag, double average, double variance) {
    double sum = 0;
    int n = nums.size();
    
    for (int i = 0; i < n - lag; i++) {
        sum += (nums[i] - average) * (nums[i + lag] - average);
    }
    
    return sum / (n - lag - 1) / variance;
}

double AutoCorrelation(std::vector<double> nums, int lag, double average) {
    double variance = Variance(nums, average);
    
    return AutoCorrelation(nums, lag, average, variance);
}

double AutoCorrelation(std::vector<double> nums, int lag) {
    double avg = Average(nums);
    
    return AutoCorrelation(nums, lag, avg);
}

#endif // STATISTICS_H

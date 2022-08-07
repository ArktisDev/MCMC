#include "metropolis.hpp"
#include "statistics.hpp"
#include "fileio.hpp"

int main() {
    std::string outputDir = "../data/";
    
    const int iterations = 1000000;
    
    std::vector<std::vector<double>> statistics;
    
    for(int stepSize = 1; stepSize <= 500; stepSize++) {
        auto res = MetropolisHastingsAlgorithm(iterations, stepSize / 100.);
        double acceptanceRatio = res.first;
        std::vector<double> samples = res.second;
        
        WriteSamplesToFile(samples, outputDir, "samples_" + std::to_string(stepSize) + ".txt");
        
        double autoCorrelation = AutoCorrelation(samples, 1);
        
        std::vector<double> row(3);
        row[0] = stepSize / 100.;
        row[1] = acceptanceRatio;
        row[2] = autoCorrelation;
        
        statistics.emplace_back(row);
    }
    
    WriteStatisticsToFile(statistics, outputDir, "statistics.txt");
}
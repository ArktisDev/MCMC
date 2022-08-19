#include "metropolis.hpp"
#include "statistics.hpp"
#include "fileio.hpp"

int main() {
    std::string outputDir = "../data/";
    
    const int iterations = 10000;
    
    std::vector<std::vector<double>> statistics;
    
    for(int stepSize = 1; stepSize <= 500; stepSize++) {
        auto res = MetropolisHastingsAlgorithm(iterations, stepSize / 100.);
        double acceptanceRatio = res.first;
        std::vector<double> samples = res.second;
        
        WriteSamplesToFile(samples, outputDir, "samples_" + std::to_string(stepSize) + ".txt");
        
        std::vector<double> row(2);
        row[0] = stepSize / 100.;
        row[1] = acceptanceRatio;

        for (int k = 1; k < 50; k++) {
            double autoCorrelation = AutoCorrelation(samples, k);
            row.emplace_back(autoCorrelation);
        }
        
        statistics.emplace_back(row);
    }
    
    WriteStatisticsToFile(statistics, outputDir, "statistics.txt");
}
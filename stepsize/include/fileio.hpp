#ifndef FILEIO_H
#define FILEIO_H

#include <string>
#include <fstream>
#include <vector>

void WriteSamplesToFile(std::vector<double> samples, std::string outputDir, std::string fileName) {
    std::ofstream ofs(outputDir + fileName);
    
    for (int i = 0; i < samples.size(); i++) {
        ofs << samples[i] << "\n";
    }
    
    ofs.flush();
    ofs.close();
}

void WriteStatisticsToFile(std::vector<std::vector<double>> statistics, std::string outputDir, std::string fileName) {
    std::ofstream ofs(outputDir + fileName);
    
    ofs << "# Stepsize Acceptance Autocorrelation\n";
    
    for (int i = 0; i < statistics.size(); i++) {
        std::vector<double> row = statistics[i];
        
        for (int j = 0; j < row.size(); j++) {
            ofs << row[j] << " ";
        }
        
        ofs <<"\n";
    }
    
    ofs.flush();
    ofs.close();
}

#endif // FILEIO_H

#ifndef INTEGRATIONOUTPUTHANDLER_H
#define INTEGRATIONOUTPUTHANDLER_H

#include "Common.cuh"

#include <vector>
#include <string>
#include <fstream>

class IntegrationOutputHandler {
  public:
    int columns;
    
    std::vector<std::vector<float>> results;
    std::vector<std::string> headers;
    
    std::string outDir;
    std::string outFile;
    
    IntegrationOutputHandler(int columns, std::vector<std::string> headers, std::string outDir, std::string outFile) 
    : columns( columns ), results( std::vector<std::vector<float>>() ), headers( headers ), outDir( outDir ), outFile( outFile )
    {
        
    }
    
    void AddRow(std::vector<float> row) {
        results.emplace_back(row);
    }
    
    void WriteToFile() {
        std::string fileName = outDir + outFile;
        
        std::ofstream ofs(fileName);
        
        ofs << "# ";
        for (const std::string& str : headers) {
            ofs << str << " ";
        }
        ofs << "\n";
        
        for (const std::vector<float>& row : results) {
            for (float entry : row) {
                ofs << entry << " ";
            }
            ofs << "\n";
        }
        
        ofs.flush();
        ofs.close();
    }
};

#endif // INTEGRATIONOUTPUTHANDLER_H

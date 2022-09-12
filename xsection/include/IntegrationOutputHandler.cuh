#ifndef INTEGRATIONOUTPUTHANDLER_H
#define INTEGRATIONOUTPUTHANDLER_H

#include "Common.cuh"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <string>

class ROOTOutputHandler {
  public:
    std::string outDir;
    std::string outFile;
    
    int nNucleonsA;
    int nNucleonsB;
    
    // Vector of 2-vectors of RMS, RMS_stderr pairs
    std::vector<std::vector<float>> rmsResults;
    bool writeRMSResults = false;
    
    // Vector of 3-vectors of b, S_AB, S_AB_stderr pairs
    std::vector<std::vector<float>> S_ABResults;
    bool writeS_ABResults = false;
    
    // Vector of 3-vectors of b, S_AB, S_AB_stderr pairs
    std::vector<std::vector<float>> I_ABResults;
    bool writeI_ABResults = false;
    
    // Vector of 3-vectors of b, S_AB, S_AB_stderr pairs
    std::vector<std::vector<float>> J_ABResults;
    bool writeJ_ABResults = false;
    
    ROOTOutputHandler(int nNucleonsA, int nNucleonsB, std::string outDir, std::string outFile) 
    : outDir( outDir ), outFile( outFile ), nNucleonsA( nNucleonsA ), nNucleonsB( nNucleonsB )
    {
        
    }
    
    void SetWriteRMSResults(bool status) {
        this->writeRMSResults = status;
    }
    
    void SetWriteS_ABResults(bool status) {
        this->writeS_ABResults = status;
    }
    
    void SetWriteI_ABResults(bool status) {
        this->writeI_ABResults = status;
    }
    
    void SetWriteJ_ABResults(bool status) {
        this->writeJ_ABResults = status;
    }
    
    void AddRMSResult(const std::vector<float>& row) {
        this->rmsResults.emplace_back(row);
    }
    
    void AddS_ABResult(const std::vector<float>& row) {
        this->S_ABResults.emplace_back(row);
    }
    
    void AddI_ABResult(const std::vector<float>& row) {
        this->I_ABResults.emplace_back(row);
    }
    
    void AddJ_ABResult(const std::vector<float>& row) {
        this->J_ABResults.emplace_back(row);
    }

    void WriteToFile() {
        TFile out((outDir + outFile).c_str(), "RECREATE");
        
        if (writeRMSResults) {
            int nucleus = -1;
            int nucleon = -1;
            float rms = -1;
            float rms_stderr = -1;
            TTree* rmsTree = new TTree("RMS", "Title");
            rmsTree->Branch("Nucleus", &nucleus, "Nucleus/I");
            rmsTree->Branch("Nucleon", &nucleon, "nucleon/I");
            rmsTree->Branch("RMS", &rms, "RMS/F");
            rmsTree->Branch("RMS_stderr", &rms_stderr, "RMS_stderr/F");
            
            for (size_t i = 0; i < rmsResults.size(); i++) {
                if (i < nNucleonsA) {
                    nucleus = 1;
                    nucleon = i + 1;
                }
                else {
                    nucleus = 2;
                    nucleon = i + 1 - nNucleonsA;
                }
                rms = rmsResults[i][0];
                rms_stderr = rmsResults[i][1];
                
                rmsTree->Fill();
            }
        }
        
        if (writeS_ABResults) {
            float b = -1;
            float S_AB = -1;
            float S_AB_stderr = -1;
            TTree* S_ABTree = new TTree("S_AB", "Title");
            S_ABTree->Branch("b", &b, "b/F");
            S_ABTree->Branch("S_AB", &S_AB, "S_AB/F");
            S_ABTree->Branch("S_AB_stderr", &S_AB_stderr, "S_AB_stderr/F");
            
            for (size_t i = 0; i < S_ABResults.size(); i++) {
                b = S_ABResults[i][0];
                S_AB = S_ABResults[i][1];
                S_AB_stderr = S_ABResults[i][2];
                
                S_ABTree->Fill();
            }
        }
        
        if (writeI_ABResults) {
            float b = -1;
            float I_AB = -1;
            float I_AB_stderr = -1;
            TTree* I_ABTree = new TTree("I_AB", "Title");
            I_ABTree->Branch("b", &b, "b/F");
            I_ABTree->Branch("I_AB", &I_AB, "I_AB/F");
            I_ABTree->Branch("I_AB_stderr", &I_AB_stderr, "I_AB_stderr/F");
            
            for (size_t i = 0; i < I_ABResults.size(); i++) {
                b = I_ABResults[i][0];
                I_AB = I_ABResults[i][1];
                I_AB_stderr = I_ABResults[i][2];
                
                I_ABTree->Fill();
            }
        }
        
        if (writeJ_ABResults) {
            float b = -1;
            float J_AB = -1;
            float J_AB_stderr = -1;
            TTree* J_ABTree = new TTree("J_AB", "Title");
            J_ABTree->Branch("b", &b, "b/F");
            J_ABTree->Branch("J_AB", &J_AB, "J_AB/F");
            J_ABTree->Branch("J_AB_stderr", &J_AB_stderr, "J_AB_stderr/F");
            
            for (size_t i = 0; i < J_ABResults.size(); i++) {
                b = J_ABResults[i][0];
                J_AB = J_ABResults[i][1];
                J_AB_stderr = J_ABResults[i][2];
                
                J_ABTree->Fill();
            }
        }
        
        out.cd();
        out.Write();
    }
};

#endif // INTEGRATIONOUTPUTHANDLER_H

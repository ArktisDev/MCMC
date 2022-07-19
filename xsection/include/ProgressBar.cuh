#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

#include <string>
#include <iostream>

class ProgressBar {
  public:
    int progress;
    int maxProgress;
    
    std::ostream* os;
    int barWidth;
    char barFullChar;
    char barEmptyChar;
    
    ProgressBar(int maxProgress, std::ostream* os = &std::cout, int barWidth = 40, char barFullChar = '*', char barEmptyChar = '-') 
    : progress( 0 ), maxProgress( maxProgress ), os( os ), barWidth( barWidth ), barFullChar( barFullChar ), barEmptyChar( barEmptyChar )
    {
        
    }
    
    void UpdateProgress(int progress) {
        this->progress = progress;
    }
    
    void IncrementProgress(int increment) {
        this->progress += increment;
    }
    
    void PrintBar() {
        int percentFilled = 100 * progress / maxProgress;
        
        *os << "\r";
        *os << "Progress: [";
        for (int i = 0; i < percentFilled * barWidth / 100; i++) {
            *os << barFullChar;
        }
        for (int i = percentFilled * barWidth / 100; i < barWidth; i++) {
            *os << barEmptyChar;
        }
        *os << "] " << percentFilled << "% " << progress << "/" << maxProgress << std::flush;
    }
    
    void FinishBar() {
        this->UpdateProgress(maxProgress);
        this->PrintBar();
        *os << "\n" << std::flush;
    }
};

#endif // PROGRESSBAR_H

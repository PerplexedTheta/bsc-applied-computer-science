// process.h -- Jake Deery, 2018
#pragma once
#ifndef process_h
#define process_h

// include my global_h
#include "global.h"

// define funcs
int16_t writeLog(stringstream& toPrint, string fileToWrite);
int16_t printVect(vector<int16_t> idVect, vector<int64_t> numVect, vector<int32_t> timeVect, string fileName);
int16_t clearVect(vector<int64_t> &srcVect);
int16_t showAvg(vector<int32_t> timeVect, vector<int16_t> avgCallsVect, string fileName);

#endif

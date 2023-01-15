// process.h -- Jake Deery, 2018
#pragma once
#ifndef process_h
#define process_h

// include my global_h
#include "global.h"

// define funcs
// int printVect(vector<int> srcVect); -- DEFUNCT :( 
int clearVect(vector<int> &srcVect);
int readFromFile(string fileToRead, vector<int> &destVect);
int writeLog(stringstream& toPrint, string fileToWrite);
int intSwap(int *in1, int *in2);
int shellSort(vector<int> &srcVect);
int insertionSort(vector<int> &srcVect);
int bubbleSort(vector<int> &srcVect);

#endif

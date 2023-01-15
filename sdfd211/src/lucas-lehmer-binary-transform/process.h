// processes.h -- Jake Deery, 2018
#pragma once
#ifndef processes_h
#define processes_h

// include my global_h
#include "global.h"

// func declarations
int16_t printVect(vector<int16_t> srcVect);
//int16_t clearVect(vector<int16_t> &destVect);
int16_t squareVect(vector<int16_t> srcVect, vector<int16_t> &destVect);
int16_t carryVect(vector <int16_t> &srcVect);
int16_t minusTwo(vector <int16_t> &srcVect);
int16_t modVect(vector<int16_t> srcVect, vector<int16_t> &destVect);

#endif

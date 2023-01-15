// main.h -- Jake Deery, 2018
#pragma once
#ifndef main_h
#define main_h

// include my h files
#include "global.h"
#include "process.h"

// func declarations
int16_t callMenu(vector<int16_t> idVect, vector<int64_t> numVect, vector<int32_t> timeVect, vector<int16_t> avgCallsVect);

// var declarations
vector<int16_t> idVect;
vector<int64_t> numVect;
vector<int32_t> timeVect;
vector<int16_t> avgCallsVect;

#endif

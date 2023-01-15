// global.h -- Jake Deery, 2018
#pragma once
#ifndef global_h
#define global_h

// include global h files
#include <iostream>
#include <ctime>
#include <chrono>
#include <sstream>
#include <vector>
#include <fstream>
#include <locale.h>
#include <math.h>

// set namespace
using std::cout;
using std::endl;
using std::cin;
using std::locale;
using std::vector;
using std::string;
using std::stringstream;
using std::fstream;
using std::ofstream;
using std::ios_base;
using uTimeGet = std::chrono::steady_clock;
using uTimeOut = std::chrono::duration<double, std::nano>;

// global vars
extern uTimeOut deltaTime;

#endif

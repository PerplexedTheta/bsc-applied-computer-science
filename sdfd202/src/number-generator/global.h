// global.h -- Jake Deery, 2018
#pragma once
#ifndef global_h
#define global_h

// define our global h files
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <locale.h>
#include <math.h>
#include <chrono>

// define namespace
using std::cout;
using std::endl;
using std::cin;
using std::locale;
using std::string;
using std::ofstream;
using uTimeGet = std::chrono::steady_clock;
using uTimeOut = std::chrono::duration<double, std::micro>;

// define global vars
extern uTimeOut deltaTime;

#endif

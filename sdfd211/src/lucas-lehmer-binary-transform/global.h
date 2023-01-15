// global.h -- Jake Deery, 2018
#pragma once
#ifndef global_h
#define global_h

// define global h files
#include <iostream>
#include <locale.h>
#include <vector>
#include <chrono>
#include <string.h>

// define namespace
using std::cout;
using std::endl;
using std::locale;
using std::cin;
using std::vector;
using uTimeGet = std::chrono::steady_clock;
using uTimeOut = std::chrono::duration<double, std::nano>;

// define int for primes
extern int16_t prime;

#endif

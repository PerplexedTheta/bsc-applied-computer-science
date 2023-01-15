// primesMenu.h -- Jake Deery 2016
#pragma once

#ifndef PRIMESMENU
#define PRIMESMENU // Defines INSTRUCTIONS

#include "mainMenu.h" // Include my mainMenu.h header file

#include "primeNumbersCalculator.h" // Includes my primeNumbersCalculator.h and primeNumbersCalculatorSieve.h header files
#include "primeNumbersCalculatorOptimised.h"
#include "primeNumbersCalculatorSieve.h"

#include "global.h" // Include my global.h and andyCode.h for this header
#include "andyCode.h"

extern int primesMenuSelected;
extern int primesMenuYLocation;
extern bool primesMenuRunning;

int primesMenu(); // This defines a function for the instructions

#endif

// extrasMenu.h -- Jake Deery 2016
#pragma once

#ifndef EXTRASMENU
#define EXTRASMENU // Define EXTRASMENU

#include "galacticStandardClock.h" // Include galacticStandardClock.h, galToLitres.h and addEvenRejectOdd.h, because it is linked to this menu
#include "galToLitres.h"
#include "processOddRejectEven.h"
#include "tenTimesForwardsBackwards.h"

#include "global.h" // Include my global.h and andyCode.h for this header
#include "andyCode.h"

extern int extrasMenuSelected;
extern int extrasMenuYLocation;
extern bool extrasMenuRunning;

int extrasMenu(); // This defines a function for the extras menu

#endif

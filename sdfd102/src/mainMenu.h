// mainMenu.h -- Jake Deery 2016
#pragma once

#ifndef MAINMENU
#define MAINMENU // Defines INSTRUCTIONS

#include "anim.h" // Includes my anim.h, extrasMenu.h, gameOfLife.h, instructions.h, primesMenu.h and randColours.h header files
#include "extrasMenu.h"
#include "gameOfLife.h"
#include "instructions.h"
#include "primesMenu.h"
#include "randColours.h"

#include "global.h" // Include my global.h and andyCode.h for this header
#include "andyCode.h"

extern int mainMenuSelected;
extern int mainMenuYLocation;
extern bool mainMenuRunning;

int mainMenu(); // This defines a function for the instructions

#endif

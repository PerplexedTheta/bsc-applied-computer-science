// global.cpp -- Jake Deery 2016
#include "global.h"

int key_pressed = 0; // Set the value of key_pressed to 0
char block = 0xdb; // Set the value of b to 219
int x = 0; // Set the value of x to 0
int y = 0; // Set the value of y to 0
DWORD clock1 = 0; // Set the value of clock1 to 0
DWORD clock2 = 0; // Set the value of clock2 to 0
LARGE_INTEGER qFrequency; // Prototype qFrequency
LARGE_INTEGER qCount1; // Prototype qCount1 
LARGE_INTEGER qCount2; // Prototype qCount2
float totaltime = 0; // Set the value of totaltime to 0
int primesCounter = 0; // Set the value of primesCounter to 0
bool menuRunning = true; // Set the value of menuRunning to true
bool entireProcessRunning = true; // Set the value of entireProcessRunning to true

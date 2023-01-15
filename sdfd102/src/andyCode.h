// andyCode.h -- Jake Deery 2016
#pragma once

#ifndef ANDYCODE
#define ANDYCODE // Define ANDYCODE

#include "global.h" // Include my global.h and andyCode.h for this header

int hline(int, int, int); // Defines hline (draw a line horizontally). Ints are: x, y, length, col/row
int vline(int, int, int); // Defines vline (draw a line vertically). Ints are: x, y, length, col/row
int gotoxy(int, int); // Defines gotoxy (moves the cursor on-screen). Ints are: x, y
int clrscr(void); // Defines clrscr (refreshes the console window).
int textcolor(int); // Sets the text colour and text bg colour (based off the RGB-256 colour table).

#endif

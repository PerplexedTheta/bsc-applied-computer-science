// global.h -- Jake Deery 2016
#pragma once

#ifndef GLOBAL
#define GLOBAL // Defines GLOBAL

#include <conio.h>
#include <iostream>
#include <Windows.h>
#include <ctime>
#include <vector>
#include <cmath>
#include <string.h>
#include <locale.h>

#define SW 80
#define SH 25
#define KEY_DOWN(vk_code) ((GetAsyncKeyState(vk_code) & 0x8000) ? 1 : 0)
#define KEY_UP(vk_code) ((GetAsyncKeyState(vk_code) & 0x8000) ? 0 : 1)
#define ESC 0x1b

extern int key_pressed; // Defines a global var for key_pressed as an int
extern int x; // Defines a global var for x as an int
extern int y; // Defines a global var for y as an int
extern DWORD clock1; // Defines a global var for clock1 as an unsigned int
extern DWORD clock2; // Defines a global var for clock2 as an unsigned int
extern LARGE_INTEGER qFrequency;
extern LARGE_INTEGER qCount1;
extern LARGE_INTEGER qCount2;
extern char block; // Defines a global var for block as a char
extern float totaltime; // Defines a global var for totaltime as a float
extern int primesCounter; // Defines a global var for primesCounter as an int
extern bool menuRunning; // Defines a global var for menuRunning as a bool
extern bool entireProcessRunning; // Define a global var for entireProcessRunning as a bool

#endif

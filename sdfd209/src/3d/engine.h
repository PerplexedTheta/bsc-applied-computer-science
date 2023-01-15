// SDFD209 -- main.h -- (C) Jake Deery, 2017
#pragma once
#ifndef engine_h
#define engine_h

// include my global header and 3dObject
#include "global.h"
#include "3dObject.h"

// identify functions
// SDFD209 -- engine.cpp -- Jake Deery, 2018
#include "engine.h"

#define _CRT_SECURE_NO_DEPRECATE

int allowedSize(int x);
void readImage(const char *filename);
void loadTexture(const char* filename);
void transform(int direction);
void display(void);
void reshape(GLsizei width, GLsizei height);
void keyboard(unsigned char key, int int1, int int2);
void special(int key, int int1, int int2);
void timer(int ms);
void initialise(void);

#endif

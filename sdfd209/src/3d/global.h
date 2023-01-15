// SDFD209 -- global.h -- (C) Jake Deery, 2017
#pragma once

#if (_MSC_VER != NULL)
	// Fix MSVC quirk
	#pragma warning(disable:4996)
#else
	// do nothing
#endif

#ifndef global_h
#define global_h

// our basic h files
#include <iostream>
#include <chrono>
#include <string.h>
#include <math.h>

// 3D or custom stuff
#include <GL/freeglut.h>

// namespace defs
using std::cout;
using std::endl;
using uTimeGet = std::chrono::steady_clock;
using uTimeOut = std::chrono::duration<double, std::nano>;

// the global header just defines the external headers that all objects need, as well as global defines
#define _CRT_SECURE_NO_WARNINGS

#define SW 1024 // NEVER CHANGE THESE!
#define SH 768
#define FPS 60

#endif

// SDFD209 -- global.h -- (C) Jake Deery, 2017
#pragma once
#ifndef global_h
#define global_h

// the global header just defines the external headers that all objects need, as well as global defines
#define SW 1024 // DONT CHANGE THESE
#define SH 768
#define FPS 60

#include <iostream>
#include <cmath>
#include <SDL2/SDL.h>
#undef main
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_mixer.h>

#endif

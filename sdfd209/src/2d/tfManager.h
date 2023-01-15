// SDFD209 -- tfManager.h -- (C) Jake Deery, 2017
#pragma once
#ifndef tfManager_h
#define tfManager_h

#include "global.h"
#include "engine.h"

class tfManager {
// public class definitions
  tfManager();
  ~tfManager();
public: 
  static SDL_Texture* loadTexture(const char* fileName);
  static int draw(SDL_Texture* tf, SDL_Rect tempR, SDL_Rect drawR);
};

#endif

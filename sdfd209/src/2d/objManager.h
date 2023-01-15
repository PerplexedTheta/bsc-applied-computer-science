// SDFD209 -- objManager.h -- (C) Jake Deery, 2017
#pragma once
#ifndef objmanager_h
#define objmanager_h

#include "global.h"
#include "engine.h"

class objManager {
// public class definitions
public:
  // first, we define the object manager
  objManager(const char* texturesheet);
  ~objManager();

  // next we define the updater and renderer. Note: the WxH and x,y co-ords are here so we can resize and move our sprite
  int update(int w, int h, int x, int y);
  int render();

//private class definitions
private:
  // set up our texture, rectangle and renderer
  SDL_Texture* objTf;
  SDL_Rect tempR, objR;
};

#endif

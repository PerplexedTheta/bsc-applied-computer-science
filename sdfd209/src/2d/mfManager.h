// SDFD209 -- mfManager.h -- (C) Jake Deery, 2017
#pragma once
#ifndef mfManager_h
#define mfManager_h

#include "global.h"
#include "engine.h"

class mfManager {
// public class definitions
public:
  mfManager(const char* musicfile, int playFlag); // define our mfManager class constructor
  ~mfManager();

  int update(); // .. and define the updater and bool
  static bool isPlaying;
private:
  static Mix_Music *loadMusic(const char* filename, int flag); // this loads our sound in for playing like in the afManager
  Mix_Music *music; // also like the afManager, this chunk is for the main sound
  Mix_Music *musF; // while this one is for the loader
};

#endif

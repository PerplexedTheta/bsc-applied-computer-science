// SDFD209 -- afManager.h -- (C) Jake Deery, 2017
#pragma once
#ifndef afManager_h
#define afManager_h

#include "global.h"
#include "engine.h"

class afManager {
  // public class definitions
public:
  afManager(const char* soundfile); // the class takes an arg, pointing to the asset file
  ~afManager();

  int update(); // updater declaration
  static bool isPlaying; // isPlaying declaration
private:
  static Mix_Chunk *loadSound(const char* filename); // this loads our sound in for playing
  Mix_Chunk *sound; // this chunk is for the main sound
  Mix_Chunk *souF; // while this one is for the loader
};

#endif

// SDFD209 -- engine.h -- (C) Jake Deery, 2017
#pragma once
#ifndef engine_h
#define engine_h

#include "global.h"
#include "tfManager.h"
#include "mfManager.h"
#include "afManager.h"
#include "objManager.h"

class Engine {
// public class definitions
public:
  // engine constructor/deconstructor
  Engine();
  ~Engine();
  
  // event handler
  int eventHandler();

  // handle key presses
  int inputHandler();
  
  //initialise the game window
  int init(const char* title, int wind_w, int wind_h, bool fullscreen);
  
  //update the game window
  int update();
  
  // main engine loop
  int engine();
  
  // clean up and exit the window
  int clean();
	
	// makes an integer its negative self
	int invInt(int input);
  
  // is the game still running?
  bool running() {
    return isRunning;
  };

  // engine clock
  int clock;

  // define a renderer name
  static SDL_Renderer* renderer;
  
// private class definitions
private:
  // sdl clock
  Uint32 startTime, finishTime;
  int deltaTime;

  // fps count
  int update_count;

  // keepalive bool
  bool isRunning;
  bool isHalted;
  
  // window manager
  SDL_Window *window;
};
#endif

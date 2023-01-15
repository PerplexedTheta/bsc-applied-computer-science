// SDFD209 -- main.cpp -- (C) Jake Deery, 2017
#include "main.h"

Engine *mainEngine = nullptr; // define mainEngine as nullptr

// initiate main program
int main(int argc, const char * argv[]) {
  //entry shit
  printf("Jake's Shit Game\nSDFD209\nBuild: 230213\n\n=====\n\n");
  printf("Engine 'mainEngine' initialising\n");

  // set our target framerate
  const int frameRate = 60;
  const int frameDelay = 1000 / frameRate; // ms between each frame

  // framerate counter
  Uint32 frameStart;
  int frameTime;

  mainEngine = new Engine(); // start a new engine
  
  mainEngine->init("Jake's Shit Game - SDFD209 - Build: 210211", 1024, 768, false);
  
  while(mainEngine->running()) {
    // detect the current tick
    frameStart = SDL_GetTicks();

    // run the game
    mainEngine->eventHandler();
    mainEngine->update();
    mainEngine->engine();

    // work out the time since the last tick
    frameTime = SDL_GetTicks() - frameStart;

    // if the frametime goes over the framedelay, pause the game in order to catch up
    // thus limiting the game to the desired framerate
    if (frameDelay > frameTime) {
      mainEngine->clock++;
      SDL_Delay(frameDelay - frameTime);
    }
  }
  
  mainEngine->clean();
  return 0;
}

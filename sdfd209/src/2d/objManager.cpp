// SDFD209 -- objManager.cpp -- (C) Jake Deery, 2017
#include "objManager.h"

// this just specifies the texture and assigns it to the renderer
objManager::objManager(const char* texturemap) {
  // define a texture for the object
  objTf = tfManager::loadTexture(texturemap);
};

objManager::~objManager() {

}

// instead of using the engine's updater and renderer, each object has it's own 'mini-engine' to lighten the workload
int objManager::update(int w, int h, int x, int y) {
  // set up the temporary rect
  tempR.w = w;
  tempR.h = h;
  tempR.x = 0;
  tempR.y = 0;

  // and now, overwrite it with the main render
  objR.w = tempR.w;
  objR.h = tempR.h;
  objR.x = x;
  objR.y = y;

  return 0;
}

// render our object
int objManager::render() {
  SDL_RenderCopy(Engine::renderer, objTf, &tempR, &objR);

  return 0;
}

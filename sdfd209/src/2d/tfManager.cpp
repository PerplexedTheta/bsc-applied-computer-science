// SDFD209 -- tfManager.cpp -- (C) Jake Deery, 2017
#include "tfManager.h"

tfManager::tfManager() {
  // unused
}

tfManager::~tfManager() {

}

// generate a texture to be used on-screen
SDL_Texture* tfManager::loadTexture(const char* tf) {
  // create a surface, attach the texture and release the surface
  SDL_Surface* tempSurface = IMG_Load(tf);
  SDL_Texture* texture = SDL_CreateTextureFromSurface(Engine::renderer, tempSurface);
  SDL_FreeSurface(tempSurface);

  printf("Texture loaded (id): %d\n", tf);

  // return the texture as the result
  return texture;
};

int tfManager::draw(SDL_Texture* tf, SDL_Rect tempR, SDL_Rect drawR) {
  SDL_RenderCopy(Engine::renderer, tf, &tempR, &drawR); // drawing the texture basically acts the same as 'render' in objManager -- it allows us to paste our updated texture to the screen

  return 0;
}

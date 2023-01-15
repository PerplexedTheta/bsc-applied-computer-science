// SDFD209 -- mfManager.cpp -- (C) Jake Deery, 2017
#include "mfManager.h"

bool mfManager::isPlaying; // define a bool for isPlaying

mfManager::mfManager(const char* musicfile, int playFlag) { // set up an object declaration
  musF = mfManager::loadMusic(musicfile, playFlag); // forward the object to the loader
}

mfManager::~mfManager() {

}

Mix_Music *mfManager::loadMusic(const char* mf, int flag) { // the music file loader. note the flag argument which allows us to loop the audio indefinitely
  Mix_Music *music = Mix_LoadMUS(mf); // define a music file using the argument

  if (music == NULL) { // if that music file doesn't load, tell the user
    printf("Music state: FAILED_LOADING\n");
    return music;
  } else if (Mix_PlayMusic(music, flag) == -1) { // if it won't play, let the user know too
    printf("Music state: FAILED_PLAYING\n");
    return music;
  }

  isPlaying = true; // finally, if it did play, set the bool and tell the user
  printf("Music loaded (id): %d\n", mf);

  return music;
}

int mfManager::update() { // the updater runs once every cycle and is used to handle 
  if (isPlaying == true && Mix_PlayingMusic() == false) { // when the music ends...
    Mix_HaltMusic(); // .. halt the music
    Mix_FreeMusic(music); // .. free the (only) music channel
    Mix_CloseAudio(); // and close the audio channel
    music = NULL; // set music to null, because unlike audio there is only one music channel

    isPlaying = false; // lastly, set isPlaying to false and alert the user
    printf("Music state: END_OF_FILE\n");
  }

  return 0;
}

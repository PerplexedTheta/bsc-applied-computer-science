// SDFD209 -- afManager.cpp -- (C) Jake Deery, 2017
#include "afManager.h"

bool afManager::isPlaying; // audio isplaying detection bool
int soundChannel; // sound channel for audio playback (max 16)

afManager::afManager(const char* soundfile) { // object manager 
  souF = afManager::loadSound(soundfile); // forward the specified asset file to the loader
}

afManager::~afManager() {

}

Mix_Chunk *afManager::loadSound(const char* af) { // load a sound, forwarded from the object manager
  Mix_Chunk *sound = NULL; // start by setting the sound int to NULL
  sound = Mix_LoadWAV(af); // next, load the audio file into sound

  if (sound == NULL) { // if loading sound failed for whatever reason, tell the user
    printf("Sound state: FAILED_LOADING\n");
    return sound;
  }

  soundChannel = Mix_PlayChannel(-1, sound, 0); // push our sound file to the playable audio channel

  isPlaying = true; // set isPlaying to true
  printf("Sound loaded (id): %d\n", af);

  return sound;
}

int afManager::update() { // audiofile updater, which checks for and releases audio files as they're done
  if (isPlaying == true && Mix_Playing(soundChannel) == 0) { // first, check to see if our sound file has reached the end
    Mix_HaltChannel(soundChannel); // if the sound has ended, halt our channel
    Mix_FreeChunk(sound); // .. and free up our sound chunk

    isPlaying = false; // lastly, set isPlaying to false
    printf("Sound state: END_OF_FILE\n");
  }

  return 0;
}

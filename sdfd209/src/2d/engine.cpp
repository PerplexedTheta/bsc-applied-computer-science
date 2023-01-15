// SDFD209 -- engine.cpp -- (C) Jake Deery, 2017
#include "engine.h"

// define a nullptr for the renderer
SDL_Renderer* Engine::renderer = nullptr;

// define an event handler
SDL_Event event;

// define a variable for our map, X&Y
int mapX = 0;
int mapY = 0;

// penguin vars, X, Y, speed, gravity, direction, jumpState, angery and render key
int penguinX = 255;
int penguinY = 768 + 251;
int penguinXmod = 0;
int penguinSpeed = 5;
float penguinVeloc = 0.0;
int penguinDirectionState = 0;
int penguinJumpState = 0;
int penguinAngeryState = 0;
int penguinRenderKey = 0;
bool penguinMoving = false;

// bear vars, X, Y and bearAngeryState
int bearX = mapX + 1124;
int bearY = SH- 166;
int bearAngeryState = 0;
int bearMovementModifier = 0;
int bearMovementDirection = 0;

// define misc -- splashstate for multiple splash screens, jumpstates for doublejumping, angerytimer for working out if penguin has annoyed bear long enough
int splashState = 0;
int jumpClock = 0;
int angeryTimer = 0;

// define our objects
objManager* splash1; // base objects
objManager* splash2;
objManager* splash3;
objManager* lvl1;

objManager* penguinR; // penguin objects
objManager* penguinRblink;
objManager* penguinRangery;
objManager* penguinL;
objManager* penguinLblink;
objManager* penguinLangery;

objManager* bearR; // enemy objects
objManager* bearRangery;
objManager* bearL; // enemy objects
objManager* bearLangery;

mfManager* music; // audio objects
afManager* boing;

// game constructor
Engine::Engine() {

}

// game deconstructor
Engine::~Engine() {

}

// initialise the game window
int Engine::init(const char *title, int wind_w, int wind_h, bool fullscreen) {
	// define things
	int wind_flags = 0;

	// if the fullscreen setting is true..
	if (fullscreen) {
	// .. make the window manager recognise this change
	wind_flags = SDL_WINDOW_FULLSCREEN;
	}

	// start the initialisation cycle
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0) {
	// if the initialisation cycle is undettered, print a state
	printf("Init state: OK\n");

	// initialise the window with all the settings applied
	window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, wind_w, wind_h, wind_flags);

	// if the initialisation cycle is undettered, print a state
	if (window) {
		printf("Window state: OK\n");
	} else {
		printf("Window state: FAILED_INIT\n");
	}

	renderer = SDL_CreateRenderer(window, -1, 0);
	// if the initialisation cycle is undettered, print a state
	if (renderer) {
		// draw a white background
		SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
		printf("Renderer state: OK\n");
	} else {
		printf("Renderer state: FAILED INIT\n");
	}

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) == 0) {
		// open a channel for our audio
		printf("Audio/video state: OK\n");
	}
	else {
		printf("Audio/video state: FAILED_INIT\n");
	}

	int audio_rate = 22050; // audiorate -- 44100 is industry standard but 22050 is healthier on low powered CPUs
	Uint16 audio_format = AUDIO_S16SYS; // set the audio format -- 16bit is industry standard
	int audio_channels = 2; // set the number of audio channels -- stereo
	int audio_buffers = 4096; // number of samples -- 4096 seems reasonable

	if (Mix_OpenAudio(audio_rate, audio_format, audio_channels, audio_buffers) != 0) { // open the audio stream
		printf("Audio/video state: FAILED_OPENING\n"); // if opening the audio/video stream failed, tell the user
	}

	// finally, if the initialisation cycle finishes fine, set isRunning to true
	isRunning = true;
	}
	else {
	// if the initialisation cycle is deterred, break the program
	printf("Init state: FAILED_INIT\n");
	isRunning = false;
	}

	// add our objects and maps -- base assets
	lvl1 = new objManager("assets/img/lvl1Bg.png");
	splash1 = new objManager("assets/img/splashBg1.png");
	splash2 = new objManager("assets/img/splashBg2.png");
	splash3 = new objManager("assets/img/splashBg3.png");

	// character assets
	penguinR = new objManager("assets/img/penguinR.png");
	penguinRblink = new objManager("assets/img/penguinRblink.png");
	penguinRangery = new objManager("assets/img/penguinRangery.png");
	penguinL = new objManager("assets/img/penguinL.png");
	penguinLblink = new objManager("assets/img/penguinLblink.png");
	penguinLangery = new objManager("assets/img/penguinLangery.png");

	// enemy assets
	bearR = new objManager("assets/img/bearR.png");
	bearRangery = new objManager("assets/img/bearRangery.png");
	bearL = new objManager("assets/img/bearL.png");
	bearLangery = new objManager("assets/img/bearLangery.png");

	// audio assets
	music = new mfManager("assets/snd/bgTrack.ogg", -1);
	//boing = new afManager("assets/snd/boing.ogg"); -- this is now found in the jump IF statement

	return 0;
}

// event handler
int Engine::eventHandler() { // handle any events here
	// while an event is happening
	while (SDL_PollEvent(&event)) {
		// process exit commands correctly
	if (event.type == SDL_QUIT) isRunning = false;
		
		inputHandler();
	}

	return 0;
}

int Engine::inputHandler() {	
	// process keydowns
	if (event.type == SDL_KEYDOWN) {
		// Q/ESCAPE
		if (event.key.keysym.sym == SDLK_q || event.key.keysym.sym == SDLK_ESCAPE) {
			isRunning = false; // kill
		}

		// ENTER
		if (event.key.keysym.sym == SDLK_RETURN && splashState < 3) { // if the splash screen is up, and enter is pressed
			splashState++; // increment said variable
		}
		
		// ENTER/P
		if ((event.key.keysym.sym == SDLK_RETURN || event.key.keysym.sym == SDLK_p) && splashState > 2) {
			isHalted = true;
		}

		// RSHIFT/LSHIFT
		if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) penguinSpeed = 9;

		// A/LEFT
		if ((event.key.keysym.sym == SDLK_a || event.key.keysym.sym == SDLK_LEFT) && splashState > 2) {
			penguinDirectionState = 1; // first, set penguin's direction to 1
			penguinXmod = invInt(penguinSpeed); // next, move back by penguinSpeed
			penguinMoving = true;

			// print co-ords
			printf("penguinX: %d\n", penguinX);
			printf("lvl1X: %d\n", mapX);
		}

		// D/RIGHT
		if ((event.key.keysym.sym == SDLK_d || event.key.keysym.sym == SDLK_RIGHT) && splashState > 2) { // D/Right key pressed and splash is gone
			penguinDirectionState = 0; // same as A/Left
			penguinXmod = penguinSpeed;
			penguinMoving = true;

			// print co-ords
			printf("penguinX: %d\n", penguinX);
			printf("lvl1X: %d\n", mapX);
		}
		
		// SPACE/W
		if ((event.key.keysym.sym == SDLK_SPACE || event.key.keysym.sym == SDLK_w) && splashState > 2) { // W/SPACE key pressed and splash is gone
			if (penguinJumpState == 0) { // if penguin is floored
				// play boing
				boing = new afManager("assets/snd/boing.ogg");

				jumpClock = clock; // match up the jumpclock and the main clock
				penguinVeloc = 5; // set penguin's jump velocity to five
				penguinJumpState = 1; // set our jumpstate to 1

				// print message
				printf("penguin jumped!\n");
			}

			if (penguinJumpState == 1 && jumpClock + 10 < clock) { // if penguin has left the floor, and the required number of frames (10) has passed
				// play boing
				boing = new afManager("assets/snd/boing.ogg");

				penguinVeloc = 5; // similar to above, excpet we have no need to sync the clocks
				penguinJumpState = 2; // push jumpstate up again

				// print message
				printf("penguin jumped!\n");
			}
		}
	}
	
	// process keydowns
	if (event.type == SDL_KEYUP) {
		// A/LEFT
		if ((event.key.keysym.sym == SDLK_a || event.key.keysym.sym == SDLK_LEFT) && splashState > 2) penguinXmod = 0;

		// D/RIGHT
		if ((event.key.keysym.sym == SDLK_d || event.key.keysym.sym == SDLK_RIGHT) && splashState > 2) penguinXmod = 0;

		// RSHIFT/LSHIFT
		if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) penguinSpeed = 5;
	}
}

//update the game engine
int Engine::update() {
	// pause the game
	while (isHalted == true) {
		if (SDL_PollEvent(&event) && event.type == SDL_KEYDOWN) { //detect keys
			if (event.key.keysym.sym == SDLK_RETURN || event.key.keysym.sym == SDLK_p) isHalted = false; // if enter or p is pressued, unpause
		} else {
			printf("game halted\np to resume\n");
		}
	}

	penguinX += penguinXmod;

	// bg updating -- A
	if (penguinX < 100) { // stop penguin from going out of bounds
		penguinX = 100;
		mapX += penguinSpeed; // .. and move the map accordingly
		bearX = mapX + 1124; // update our bear location so he comes onscreen
	}

	// check map bounds -- A
	if (mapX > 0) { // if map is over 0
		mapX = -2048; // bounce here
	}

	// bg updating -- D
	if (penguinX > (SW - (251 + 100))) { // same as above
		penguinX = (SW - (251 + 100));
		mapX -= penguinXmod;
		bearX = mapX + 1124;
	}

	// check map bounds -- D
	if (mapX < (-3072 + SW)) { // same as above
		mapX = 0;
	}

	// if pingu isnt moving, we want to make our bear pace back and forth
	if (penguinMoving == false && bearMovementDirection == 0 && clock % 12 == 0) { // notice how the clock has to be a modulo of 12 -- aka every 12 frames -- and the bear must face direction 0
		if (bearMovementModifier < 5) { // simply put, when the bear is walking and our movement modifier is less than 5
			bearX++; // make bear move up the map
			bearMovementModifier++; // and increment the modifier
		} else { // once our modifier goes out of range, reset the modifier to 0 and change direction
			bearMovementModifier = 0;
			bearMovementDirection = 1;
		}
	}

	if (penguinMoving == false && bearMovementDirection == 1 && clock % 12 == 0) { // same as above, except bear goes down the map
		if (bearMovementModifier < 5) {
			bearX--;
			bearMovementModifier++;
		} else {
			bearMovementModifier = 0;
			bearMovementDirection = 0;
		}
	}

	// then detect if it is bound
	if (penguinX + 251 > bearX && // if penguin has passed the left edge
	penguinX < bearX + 256 && // if penguin has passed the right edge
	penguinY + 255 > bearY && // if penguin is below the top edge
	penguinY < bearY + 166){ // if penguin is above the bottom -- the entire rectangle detected in one IF statement. (sunglasses)

	bearAngeryState = 1;

	if (angeryTimer == 0) { // the angery timer basically detects how long penguin and bear have been colided
		angeryTimer = clock;
	}

	if (penguinAngeryState == 1) {
		// print to console -- GAME SUCCESS
		printf("\npenguin collided with bear!\n");
		printf("penguin was ready for a fight!\n\n");
		printf("===============\n");
		printf("= PLAYER WINS =\n");
		printf("===============\n\n");

		// kill
		isRunning = false;
	} else if (penguinAngeryState == 0 && angeryTimer + 120 < clock) {
		// print to console -- GAME OVER
		printf("\npenguin collided with bear!\n");
		printf("penguin sat on the bear too long without getting angry!\n\n");
		printf("=============\n");
		printf("= GAME OVER =\n");
		printf("=============\n\n");

		// kill
		isRunning = false;
	} else {
		// print to console -- GAME OVER
		printf("penguin collided with bear!\n");
		printf("get ready to fight!\n");
	}
	} else { // if none of these conditions are reached, reset the bear's anger and anger timer
	angeryTimer = 0; 
	bearAngeryState = 0;
	}

	// draw base
	splash1->update(SW, SH, 0, 0);
	splash2->update(SW, SH, 0, 0);
	splash3->update(SW, SH, 0, 0);
	lvl1->update((3 * SW), SH, mapX, mapY);

	// update our objects
	penguinR->update(251, 255, penguinX, penguinY); // each update takes in the width, height, xpos and ypos based on previous updates
	penguinRblink->update(251, 255, penguinX, penguinY); // in my examples, I have a static WxH (251,255) and dynamic XxY
	penguinRangery->update(251, 255, penguinX + (rand() % 6), penguinY + (rand() % 6)); // penguinX and penguinY are altered above based on jumping and moving L/R
	penguinL->update(251, 255, penguinX, penguinY);
	penguinLblink->update(251, 255, penguinX, penguinY);
	penguinLangery->update(251, 255, penguinX + (rand() % 6), penguinY + (rand() % 6)); // the rand modulo 6 makes our penguin shake with anger a bit

	bearR->update(256, 166, bearX, bearY); // update our bear object
	bearRangery->update(256, 166, bearX + (rand() % 6), bearY + (rand() % 6)); // the rand modulo 6 makes our bear shake with anger a bit
	bearL->update(256, 166, bearX, bearY); // same as above
	bearLangery->update(256, 166, bearX + (rand() % 6), bearY + (rand() % 6));


	music->update(); // audio updaters take no arguments -- they are simply sensors
	boing->update();

	// work out our current framerate and NOW clock value
	update_count++;

	// detect a start time, and a "deltaTime", i.e. the difference between a and b
	startTime = SDL_GetTicks();
	deltaTime = (startTime - finishTime);

	if (deltaTime > 1000) {
	// print a framerate to console
	printf("framerate: %d\n", update_count);
	printf("total frames: %d\n", clock);

	// get finishTime
	finishTime = SDL_GetTicks();

	// reset update_count
	update_count = 0;
	}

	// limit penguin to the floor and set jumpState
	if (penguinY > SH - 255) {
	penguinY = SH - 255;
	if (penguinJumpState == 1 || penguinJumpState == 2) { // if jumping once, or double-jumping
		penguinJumpState = 0; // not jumping
	}
	}

	// make penguin jump! -- updater
	penguinVeloc -= 0.2; // increase fall speed
	if (penguinVeloc < -5.0) { // terminal velocity
	penguinVeloc = -5.0;
	}
	penguinY -= penguinVeloc; // make penguin fall

	return 0;
}

// render each frame
int Engine::engine() {
	// to start the loop: clear the renderer
	SDL_RenderClear(renderer);

	if (splashState == 0) { // if splashState is at position 1
	splash1->render();
	} else if (splashState == 1) { // if splashState is at position 2
	splash2->render();
	} else if (splashState == 2) { // if splashState is at position 3
	splash3->render();
	} else {

	lvl1->render(); // draw map 1

	// bear select
	if (bearMovementDirection == 0) {
		if (bearAngeryState == 0) { // bearR
		bearR->render();
		}

		if (bearAngeryState == 1) { // bearRangery
		bearRangery->render();
		}
	}

	if (bearMovementDirection == 1) {
		if (bearAngeryState == 0) { // bearL
		bearL->render();
		}

		if (bearAngeryState == 1) { // bearLangery
		bearLangery->render();
		}
	}
	
	if (clock % 180 == 0) { // set a render key for penguin's blinking
		penguinRenderKey = 6; // every 3s, set penguin's render key to 6
	}

	// penguin select
	if (penguinAngeryState == 0) {
		if (penguinDirectionState == 0) { // penguinR
		if (penguinRenderKey > 0) { // if the render key is more than 0
			penguinRblink->render(); // render blinky
			penguinRenderKey--; // de-increment render key
		}

		if (penguinRenderKey == 0) { // if the render key is 0
			penguinR->render(); // render normal penguin
		}
		}

		if (penguinDirectionState == 1) { // penguinL
		if (penguinRenderKey > 0) { // same as above
			penguinLblink->render();
			penguinRenderKey--;
		}

		if (penguinRenderKey == 0) { // same as above
			penguinL->render();
		}
		}
	}

	if (penguinAngeryState == 1) {
		if (penguinDirectionState == 0) { // penguinRangery
		penguinRangery->render();
		}

		if (penguinDirectionState == 1) { // penguinLangery
		penguinLangery->render();
		}
	}
	}

	// to finish the loop: render the contents of the loop
	SDL_RenderPresent(renderer);

	return 0;
}

// clean up and exit the window
int Engine::clean() {
	SDL_DestroyWindow(window); // destroy the window
	SDL_DestroyRenderer(renderer); // kill the render process
	SDL_CloseAudio(); // kill the audio process
	SDL_Quit(); // quit SDL and release memory
	printf("\nEngine state: 1\n");

	return 0;
}

// misc functions
int Engine::invInt(int input) {
	int output = 0;
	
	output = input * -1;
	return output;
}

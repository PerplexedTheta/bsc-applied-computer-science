// randColours.cpp -- Jake Deery 2016
#include "randColours.h"

int colours = 0;

int randColours() { // Function for some random colours
	clrscr(); // Clear the screen to start with

	do {
		//COMMENT NEXT LINE TO SET TO BLOCKS
		block = 97 + (rand() % 25); // This line essentially randomises char b: 97 is the minimum, 25 is max - min. This gives a range of 97-122
		x = rand() % SW; // Randomise a row on the screen, modular of SW
		y = rand() % SH; // Randomise a column on the screen, modular of the SH
		colours = rand() % 16; // Randomise colours between 0-15.
		textcolor(colours); // Sets the text colour to match the value set above
		gotoxy(x, y); // Go to the values of x and y
		std::cout << block; // Print out a character matching the current value of b
	} while (!KEY_DOWN(VK_ESCAPE));

	mainMenu();
	return 0;
}

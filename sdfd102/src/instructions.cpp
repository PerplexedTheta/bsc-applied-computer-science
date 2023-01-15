// instructions.cpp -- Jake Deery 2016
#include "instructions.h"

int instructions() { // Commence main process
	clrscr(); // Clear the screen to begin

	std::cout << "Build number: 2017,8,14,1";

	textcolor(13); // Set the colour of the lines
	hline(1, 1, 78); // Top hline
	vline(78, 2, 26); // Right vline
	hline(1, 27, 78); // Bottom hline
	vline(1, 2, 26); // Left vline

	textcolor(10); // Set the text colour to green
	gotoxy(3, 3); // Move the cursor to position [3,3]
	std::cout << "Welcome to Jake's sexi menu! -- Instructions"; // Print this line
	textcolor(14); // Set the colour to yellow
	gotoxy(3, 5);
	std::cout << "* On the menus, use the (UP) and (DOWN) arrows to move the pointer to the";
	gotoxy(3, 6);
	std::cout << "program you wish to run, then press (RETURN).";
	gotoxy(3, 8);
	std::cout << "* Controls for animation: Use your arrow keys to move the jar around the";
	gotoxy(3, 9);
	std::cout << "screen. Press (ESC) to exit.";
	gotoxy(3, 10);
	std::cout << "* Controls for random characters: Press (ESC) to exit.";
	gotoxy(3, 11);
	std::cout << "* Controls for Game of Life: Press (G) to make a gosper glider gun or";
	gotoxy(3, 12);
	std::cout << "(SPACEBAR) to reset the game. Press (ESC) to exit.";
	gotoxy(3, 14);
	std::cout << "* Controls for prime numbers calculators: Press any key to exit.";
	gotoxy(3, 16);
	std::cout << "* Controls for galactic standard clock: Press (R) to reset. Press (ESC) to";
	gotoxy(3, 17);
	std::cout << "exit.";
	gotoxy(3, 18);
	std::cout << "* Controls for a gallons to litres calculator: Enter a random number";
	gotoxy(3, 19);
	std::cout << "and press (RETURN). From there, either press any key to repeat or press";
	gotoxy(3, 20);
	std::cout << " (ESC) to exit.";
	gotoxy(3, 21);
	std::cout << "* Control for add odd, reject even calculator: Enter a random number.";
	gotoxy(3, 22);
	std::cout << "Once the last input number is even, press any key to exit.";
	gotoxy(3, 23);
	std::cout << "* Controls for ten times, forwards and backards: Press any key to exit.";
	textcolor(10); // Set the colour to green again
	gotoxy(3, 25);
	std::cout << "Press any key to continue . . ."; // Ask the user to exit
	gotoxy(0,0); // Reset the cursor position
	system("pause>nul");

	mainMenu();
	return 0;
}

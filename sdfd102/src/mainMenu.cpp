// mainMenu.cpp -- Jake Deery 2016
#include "mainMenu.h"

int mainMenuSelected = 0;
int mainMenuYLocation = 5;

int mainMenu() { // Commence main process
	menuRunning = true;

	block = 219;
	clrscr();

	textcolor(13); // Set the colour of the lines
	hline(1, 1, 78); // Top hline
	vline(78, 2, 22); // Right vline
	hline(1, 23, 78); // Bottom hline
	vline(1, 2, 22); // Left vline

	textcolor(14); // Set the colour of the top line of text (yellow)
	gotoxy(4, 3); // Move the cursor to position [4,3]
	std::cout << "Welcome to Jake's sexi menu!"; // Print this text...
	textcolor(10); // Set the colour of the top line of text (green)
	gotoxy(4, 5);
	std::cout << "(1) See the instructions";
	gotoxy(4, 6);
	std::cout << "(2) See an animation";
	gotoxy(4, 7);
	std::cout << "(3) See some sexi colours";
	gotoxy(4, 8);
	std::cout << "(4) Play a Game of Life";
	gotoxy(4, 9);
	std::cout << "(5) Run some fast prime numbers";
	gotoxy(4, 10);
	std::cout << "(6) See the extras menu";
	gotoxy(4, 11);
	std::cout << "(E) Quit Program";
	textcolor(14); // Set the colour of the top line of text (yellow)
	gotoxy(4, 13);
	std::cout << "Press (UP) or (DOWN) and then press (RETURN).";
	textcolor(10); // Set the colour of the top line of text (green)
	gotoxy(2, mainMenuYLocation); // On this line, we move the cursor to the 2nd column, and to the row relating to the current selector location.
	std::cout << "->";

	while (menuRunning) { // While menuRunning is true...
		gotoxy(0,0); // Reset the cursor position
		system("pause>nul");
		
		if (KEY_DOWN(VK_DOWN) && mainMenuYLocation != 11) { // If the user pressed the down arrow, and the mainMenuYLocation is not 11...
			gotoxy(2, mainMenuYLocation); // Gotoxy where 2 is the column and mainMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			mainMenuYLocation++; // Incriment mainMenuYLocation by one
			gotoxy(2, mainMenuYLocation); // Gotoxy where 2 is the column and mainMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			mainMenuSelected++; // Incriment the mainMenuSelected by one
		}
			
		if (KEY_DOWN(VK_UP) && mainMenuYLocation != 5) { // If the user pressed the up arrow, and the mainMenuYLocation is not 5...
			gotoxy(2, mainMenuYLocation); // Gotoxy where 2 is the column and mainMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			mainMenuYLocation--; // Decrese mainMenuYLocation by one
			gotoxy(2, mainMenuYLocation); // Gotoxy where 2 is the column and mainMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			mainMenuSelected--; // Decrease mainMenuSelected by one
		}
			
		if (KEY_DOWN(VK_RETURN)) { // If the user pressed the return key...
			switch (mainMenuSelected) { // Make a switch, that reads from mainMenuSelected
				case 0: { // If mainMenuSelected = 0...
					instructions(); // Show the instructions
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 1: { // If mainMenuSelected = 1...
					anim(); // Run the anim function
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 2: { // If mainMenuSelected = 2...
					randColours(); // Run the randColours function
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 3: { // If mainMenuSelected = 3...
					gameOfLife(); // Run the gol function
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 4: { // If mainMenuSelected = 4...
					primesMenu(); // Run the primesMenu function
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 5: { // If mainMenuSelected = 5...
					extrasMenu(); // Run the extraMenu function
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 6: { // If mainMenuSelected = 6...
					clrscr(); // Clear the screen...
					gotoxy(0, 0); // Gotoxy where the column is 0 and the row is 0 too
					std::cout << "The process ended sucessfuly!"; // Print out this line...
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					exit (EXIT_SUCCESS); // Kill the program (NAUGHTY METHOD)
				}
			}
		}
	}

	return 0;
}

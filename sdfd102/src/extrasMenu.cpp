// extrasMenu.cpp -- Jake Deery 2016
#include "extrasMenu.h"

int extrasMenuSelected = 0;
int extrasMenuYLocation = 5;

int extrasMenu() { // Commence main process
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
	std::cout << "Welcome to Jake's sexi menu! -- Extras menu"; // Print this text...
	textcolor(10); // Set the colour of the top line of text (green)
	gotoxy(4, 5);
	std::cout << "(1) Start a galactic clock.";
	gotoxy(4, 6);
	std::cout << "(2) Convert gallons to litres.";
	gotoxy(4, 7);
	std::cout << "(3) Run an odd-only calculator.";
	gotoxy(4, 8);
	std::cout << "(4) Run the 10x table forwards and backwards.";
	gotoxy(4, 9);
	std::cout << "(E) Exit menu";
	textcolor(14); // Set the colour of the top line of text (yellow)
	gotoxy(4, 11);
	std::cout << "Press (UP) or (DOWN) and then press (RETURN).";
	textcolor(10); // Set the colour of the top line of text (green)
	gotoxy(2, extrasMenuYLocation); // On this line, we move the cursor to the 2nd column, and to the row relating to the current selector location.
	std::cout << "->";
		
	while (menuRunning) { // While menuRunning is true...
		gotoxy(0,0); // Reset the cursor position
		system("pause>nul");
		
		if(KEY_DOWN(VK_DOWN) && extrasMenuYLocation != 9) { // If the user pressed the down arrow, and the extrasMenuYLocation is not 9...
			gotoxy(2, extrasMenuYLocation); // Gotoxy where 2 is the column and extrasMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			extrasMenuYLocation++; // Incriment extrasMenuYLocation by one
			gotoxy(2, extrasMenuYLocation); // Gotoxy where 2 is the column and extrasMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			extrasMenuSelected++; // Incriment the extrasMenuSelected by one	
		}
			
		if(KEY_DOWN(VK_UP) && extrasMenuYLocation != 5) { // If the user pressed the up arrow, and the extrasMenuYLocation is not 5...
			gotoxy(2, extrasMenuYLocation); // Gotoxy where 2 is the column and extrasMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			extrasMenuYLocation--; // Decrese extrasMenuYLocation by one
			gotoxy(2, extrasMenuYLocation); // Gotoxy where 2 is the column and extrasMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			extrasMenuSelected--; // Decrease extrasMenuSelected by one
		}
			
		if(KEY_DOWN(VK_RETURN)) { // If the user pressed the return key...
			switch(extrasMenuSelected) { // Make a switch, that reads from extrasMenuSelected
				case 0: { // If extrasMenuSelected = 0...
					galacticStandardClock(); // Run the galactic standard clock
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 1: { // If extrasMenuSelected = 1...
					galToLitres(); // Run the gallons to litres calculator
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 2: { // If extrasMenuSelected = 2...
					processOddRejectEven(); // Run the add odd, reject even calculator
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 3: { // If extrasMenuSelected = 3...
					tenTimesForwardsBackwards(); // Run the add even, reject odd calculator
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 4: { // If extrasMenuSelected = 4...
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				}
			}
		}		
	}

	mainMenu();
	return 0;
}

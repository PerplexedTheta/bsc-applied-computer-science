// primesMenu.cpp -- Jake Deery 2016
#include "primesMenu.h"

int primesMenuSelected = 0;
int primesMenuYLocation = 5;

int primesMenu() { // Commence main process
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
	std::cout << "Welcome to Jake's sexi menu! -- Primes menu"; // Print this text...
	textcolor(10); // Set the colour of the top line of text (green)
	gotoxy(4, 5);
	std::cout << "(1) Calculate Prime Numbers with Bruteforce.";
	gotoxy(4, 6);
	std::cout << "(2) Calculate Prime Numbers, Optimised";
	gotoxy(4, 7);
	std::cout << "(3) Calculate Prime Numbers with Sieve of Eratosthenes.";
	gotoxy(4, 8);
	std::cout << "(E) Exit menu";
	textcolor(14); // Set the colour of the top line of text (yellow)
	gotoxy(4, 10);
	std::cout << "Press (UP) or (DOWN) and then press (RETURN).";
	textcolor(11); // Set the colour of the top line of text (green)
	gotoxy(2, primesMenuYLocation); // On this line, we move the cursor to the 2nd column, and to the row relating to the current selector location.
	std::cout << "->";
		
	while (menuRunning) { // While menuRunning is true...
		gotoxy(0,0); // Reset the cursor position
		system("pause>nul");
		
		if(KEY_DOWN(VK_DOWN) && primesMenuYLocation != 8) { // If the user pressed the down arrow, and the mainMenuYLocation is not 9...
			gotoxy(2, primesMenuYLocation); // Gotoxy where 2 is the column and primesMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			primesMenuYLocation++; // Incriment primesMenuYLocation by one
			gotoxy(2, primesMenuYLocation); // Gotoxy where 2 is the column and primesMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			primesMenuSelected++; // Incriment the primesMenuSelected by one
		}
			
		if(KEY_DOWN(VK_UP) && primesMenuYLocation != 5) {
			gotoxy(2, primesMenuYLocation); // Gotoxy where 2 is the column and primesMenuYLocation is the row
			std::cout << "  "; // Set whatever was there before to a blank space
			primesMenuYLocation--; // Decrease primesMenuYLocation by one
			gotoxy(2, primesMenuYLocation); // Gotoxy where 2 is the column and primesMenuYLocation is the row
			std::cout << "->"; // Print out an arrow to tell the user where the menu selector is.
			primesMenuSelected--; // Decrease the primesMenuSelected by one
		}
			
		if(KEY_DOWN(VK_RETURN)) { // If the user pressed the return key...
			switch(primesMenuSelected) { // Make a switch, that reads from primesMenuSelected
				case 0: {
					primeNumbersCalculator(); // Run some prime nos
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 1: {
					primeNumbersCalculatorOptimised(); // Run some primes nos (optimised)
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 2: {
					primeNumbersCalculatorSieve(); // Run the sieve calculator
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				} case 3: {
					menuRunning = false; // Set the menuRunning to false (so the while loop deactivates)
					break; // And break the loop too
				}
			}
		}		
	}
	
	mainMenu();
	return 0;
}

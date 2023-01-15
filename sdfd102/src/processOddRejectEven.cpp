// processOddRejectEven.cpp -- Jake Deery 2016
#include "processOddRejectEven.h"

bool isOdd(int); // Define a bool used for true/false

int input; // Define the variables for this program
int output;
bool processOddRejectEvenRunning = true;

int processOddRejectEven() { // Commence main process
	clrscr(); // Clear the screen!
	output = 0;
	processOddRejectEvenRunning = true;

	std::cout << "Process odd numbers, reject even numbers." << std::endl; // Print the instructions...
	std::cout << "Enter an integer." << std::endl;
	std::cout << "Enter an even number to end the program." << std::endl << std::endl;

	do { // Do while processOddRejectEvenRunning is true
		std::cout << "Please enter an integer: "; // Ask the user to enter an int
		std::cin >> input; // Ask the user to enter a value into input
		
		output = output + input; // Basically, make output the value of its former self, plus whatever you put in input

		if (isOdd(input) == true) { // IfElse cycle: if the value of output is odd... else... 
			std::cout << output << " is the current sum total." << std::endl << std::endl; // This basically just prints the current value of output
		}
		else { // This section runs when the value of output is even
			std::cout << output << " is the final sum total!" << std::endl << std::endl; // This basically just prints the current value of output
			processOddRejectEvenRunning = false; // ... and this ends the program for good.
		}
	} while (processOddRejectEvenRunning);
	std::cout << "Press any key to continue . . ."; // This asks the user to end the program
	system("pause>nul");

	extrasMenu();
	return 0;
}

bool isOdd(int integer) { // Commence bool process
	if (integer % 2 == 0) // This IfElse cycle basically states that if the number is a modular of two, it's a false bool, else it's true.
		return false;
	else
		return true;
}

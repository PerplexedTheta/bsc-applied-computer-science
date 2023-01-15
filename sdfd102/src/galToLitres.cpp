// galToLitres.cpp -- Jake Deery 2016
#include "galToLitres.h"

const float oneLitreInGallons = 4.546090; // Define a constant float, oneLitreInGallons (which stores the value of what 1 litre is in gallons), and two empty floats named input and outputs.
float input;
float output;

int galToLitres() {
	clrscr(); // To begin with, clear the screen.

	std::cout << "Gallons (UK) to Litres converter." << std::endl; // Program title
	std::cout << "Enter an integer." << std::endl; // Instructions
	std::cout << "Please run the program at least once, and then press (ESC) to exit." << std::endl << std::endl; // Exit procedure

	do { // Next, open a do/while loop that repeats the process each time the process exits successfully
		std::cout << "Please enter a number to of gal(UK) convert: "; // Now, ask the user to input a value in gallons
		std::cin >> input; // ... and actually put the input into (float)input

		output = ((float)input*(float)oneLitreInGallons); // On this line, we do the calculation: input x 4.546090 = output

		std::cout << "That is " << output << " litres." << std::endl << std::endl; // On this line, we print the result of the calculation on the screen

		std::cout << "Press any key to continue OR press (ESC) to exit . . ." << std::endl; // Finally, tell the user to press any key to either repeat or exit (ESC)
		key_pressed = _getch(); // Wait for input, placing the result in key_pressed
	} while (key_pressed != ESC); // While Key_pressed is not ESC, keep running the process. Else, exit...

	extrasMenu();
	return 0;
}

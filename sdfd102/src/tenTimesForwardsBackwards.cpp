// tenTimesForwardsBackwards.cpp -- Jake Deery 2016
#include "tenTimesForwardsBackwards.h"

int iterator; // Set multiplier to value 1 at the start
int tenTimesMultiplier = 10; // Set tenTimes to value 10
int total; // Define totla as an int

int tenTimesForwardsBackwards() { // Commence main process
	clrscr();

	std::cout << "Ten times table forwards, and backwards." << std::endl; // Program title
	std::cout << "Press any key to exit the program." << std::endl << std::endl; // Exit procedure

	iterator = 0;
	total = 0;

	while (total<120) { // While total is less than 120, do this loop
		total = iterator*tenTimesMultiplier; // total will equal the value of iterator times the value of tenTimesMultiplier
		std::cout << total << std::endl; // Print the current value of total
		iterator++; // Incriment iterator by one
	}

	while (total != 0) { // Once the other loop ends, start this loop, where total should value other than 0
		total = total - tenTimesMultiplier; // This will basically cut 10 from the value on each loop
		std::cout << total << std::endl; // Print the current value of total
	}

	std::cout << "Press any key to continue . . ."; // Asks the user to exit
	system("pause>nul");

	extrasMenu();
	return 0;
}

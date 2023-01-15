// main.cpp -- Jake Deery 2016
/* ------------------------------------------- */
/* SDFD102 Final Assignment -- Jake Deery 2016 */
/* ------------------------------------------- */
#include "main.h" // Each CPP file has it's own .h file

int main() { // ... and it's own function.
		// set the locale for the program
	std::cout.imbue(std::locale(""));

	do {
		instructions(); // Call upon instructions();
	} while(entireProcessRunning);
	return 0;
}

// primeNumbersCalculator.cpp -- Jake Deery 2016
#include "primeNumbersCalculator.h"

int primeNumbersCalculator()
{
	clrscr(); // Clear the screen to begin
	
	std::cout << "Prime Numbers Calculator: Bruteforce to " << 100000 << std::endl; // Program title
	std::cout << "Press any key to continue . . ." << std::endl; // Ask the user to wait
	system("pause>nul");
	std::cout << std::endl << "WARNING: This is going to take a while... Please wait!" << std::endl << std::endl; // Warn the user this could take some time

	primesCounter = 0;
	
	QueryPerformanceFrequency(&qFrequency); // This basically detects how good the current version of windows is at keeping time
	QueryPerformanceCounter(&qCount1); // Begin the timer
	
	for (int numberTesting = 3; numberTesting < 100000; numberTesting++) { // This is the main number testing loop. Basically, so long as 100,000 hasn't been reached, it'll keep going.
		int primeInt = 1; // To begin with, define prime as a true int

		for (int modularTesting = 3; modularTesting < numberTesting; modularTesting++) { // In this loop, we have a modular tester, which basically keeps running so long as modular tester is less than the number being tested
			if (fmod(long double(numberTesting), long double(modularTesting)) == 0) {	// Once the modularTester has been set, we need to work out if it's the modular of numberTesting. If it isn't...
				primeInt = 0; // ... set prime to false...
				break; // ... and break the loop
			}
		}

		if (primeInt == 1) { // Basically, if the prime bool is true... 
			primesCounter++; // ... increment the primes counter by one...
			// Comment this line to stop printing
			std::cout << numberTesting << " "; // ... and print the number being tested as a prime
		}
	}
	QueryPerformanceCounter(&qCount2); // End the timer
	
	totaltime = (long float)(((long float)qCount2.QuadPart - (long float)qCount1.QuadPart) * 100000 / qFrequency.QuadPart) / 100000; // Make totaltime	the difference between qCount2 and qCount1...
	std::cout << std::endl << std::endl << "Total time (sec): " << totaltime << std::endl; // ... print ticktotal...
	std::cout << primesCounter << " primes found." << std::endl; // ... and the number of primes found...

	std::cout << "Press any key to continue . . ."; // This asks the user to end the program
	system("pause>nul");

	primesMenu();
	return 0;
}

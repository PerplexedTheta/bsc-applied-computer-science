// primeNumbersCalculatorOptimised.cpp -- Jake Deery 2016
#include "primeNumbersCalculatorOptimised.h"

std::vector<int> primeNumbers;

int primeNumbersCalculatorOptimised() {
	clrscr(); // Clear the screen to begin
	
	std::cout << "Prime Numbers Calculator: Bruteforce to " << 100000 << ", Optimised" << std::endl; // Program title
	std::cout << "Press any key to continue . . ." << std::endl; // Ask the user to wait
	system("pause>nul");
	std::cout << std::endl << "WARNING: This is going to take a while... Please wait!" << std::endl << std::endl; // Warn the user this could take some time

	primeNumbers.push_back(2); // Push the prime numbers vector back by two

	QueryPerformanceFrequency(&qFrequency); // This basically detects how good the current version of windows is at keeping time
	QueryPerformanceCounter(&qCount1); // Begin the timer
	
	for (int numberTesting = 3; numberTesting < 100000; numberTesting++) { // Within this for loop, starting from 3, we count up to 100,000 in steps of one. This number will be the actual number we test.
		bool prime = true; // To start with, set the 'prime' bool to true. This bool will be used to determine whether or not the calculated number is prime.

		for (int modularTester = 0; modularTester < primeNumbers.size() && primeNumbers[modularTester] * primeNumbers[modularTester] <= numberTesting; modularTester++) { // Within this for loop, we essentially define a modular tester which will go through and incriment the modularTester if it is a) less than the total size of the primeNumbers vector and b) the sqrt of modularTester within the vector is lte numberTesting
			if (numberTesting % primeNumbers[modularTester] == 0) { // Once the modularTester has been set, we need to work out if it's the modular of numberTesting. If it isn't...
				prime = false; // ... set prime to false...
				break; // ... and break the loop
			}
		}

		if (prime)  { // ... if the result from the previous loop wasn't false...
			primeNumbers.push_back(numberTesting); // ... add the value passed to the end of the vector...
			// Comment this line to stop printing
			std::cout << numberTesting << " "; // ... and print the number being tested as a prime
		}
	}
	QueryPerformanceCounter(&qCount2); // End the timer
	
	totaltime = (long float)(((long float)qCount2.QuadPart - (long float)qCount1.QuadPart) * 100000 / qFrequency.QuadPart) / 100000; // Make totaltime	the difference between qCount2 and qCount1...
	std::cout << std::endl << std::endl << "Total time (sec): " << totaltime << std::endl; // ... print ticktotal...
	std::cout << primeNumbers.size() << " primes found." << std::endl; // ... and the number of primes found...
	primeNumbers.clear();

	std::cout << "Press any key to continue . . ."; // This asks the user to end the program
	system("pause>nul");

	primesMenu();
	return 0;
}

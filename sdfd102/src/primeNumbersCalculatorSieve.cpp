// primeNumbersCalculatorSieve.cpp -- Jake Deery 2016
#include "primeNumbersCalculatorSieve.h"

const int numbersToTest = 100000; // This is the number we want to end at
int sqrtPrime = sqrt(long double(numbersToTest)); //This is the square root of the end number (i.e. 100,000)
bool * arrayOfPrimes = new bool[numbersToTest]; // This creates an array of bools. Bools are useful because they are only 1 bit in size.
int primeMultiplier = 0; // This creates an int where we multiply the result from both for loops and work out if we have a prime or not
int counter = 0; // This counter stores the number of primes we have calculated in the process

int primeNumbersCalculatorSieve() {
	// set the locale for the program
	setlocale(LC_NUMERIC, "el_GR");

	clrscr();

	std::cout << "Prime Numbers Calculator: Sieve of Eratosthenes (To " << 100000 << ")" << std::endl;
	std::cout << "Press any key to continue . . ." << std::endl;
	system("pause>nul");
	std::cout << std::endl;

	for (int arrayIterator = 3; arrayIterator <= numbersToTest; ++arrayIterator) { // This loop simply iterates up to the number we want to end at
		arrayOfPrimes[arrayIterator] = arrayIterator % 2; // This will fill the array of primes with bools, half odd and half even
	}

	primeMultiplier = 0; // To start with, reset the prime multiplier (explained below) incase the program is being rerun
	
	QueryPerformanceFrequency(&qFrequency); // This basically detects how good the current version of windows is at keeping time
	QueryPerformanceCounter(&qCount1); // Begin the timer

	for (short iterator1 = 3; iterator1 < sqrtPrime; iterator1 += 2) { // Firstly, we run through and work out all the square roots
		if (arrayOfPrimes[iterator1]) { // Basically, this if statement saves us time by not having to look at the same numbers twice (checks if iterator1 is true basically)
			for (int iterator2 = 3; iterator2 < (numbersToTest/3); iterator2 += 2) { // This loop basically goes through and multiplies the first iterator with the second
				primeMultiplier = iterator1 * iterator2; // Here is where the multiplication happens
				if (primeMultiplier > numbersToTest) { // This basically checks to see if the multiplied value is higher than the maximum value
					break; // ... and if it is, break the for loop
				}
				arrayOfPrimes[primeMultiplier] = false; // If the for loop is not broken above, this line will make the number not a prime in the bool array
			}
		}
	}

	QueryPerformanceCounter(&qCount2); // Kill the timer

	counter = 1;
	for (int arrayCountUp = 0; arrayCountUp <= numbersToTest; ++arrayCountUp) {
		if (arrayOfPrimes[arrayCountUp] == true) { // This basically goes through and counts up all the true values in the array (aka the primes)
			counter++;
		}
	}
	
	totaltime = (long float)(((long float)qCount2.QuadPart - (long float)qCount1.QuadPart) * 100000 / qFrequency.QuadPart) / 100000; // Make totaltime the difference between qCount2 and qCount1
	std::cout << "Total time (sec): " << totaltime << std::endl; // Print totaltime
	std::cout << counter << " primes found." << std::endl; // ... and the number of primes found...

	std::cout << "Press any key to continue . . ." << std::endl; // This asks the user to end the program
	system("pause>nul");

	primesMenu();
	return 0;
}

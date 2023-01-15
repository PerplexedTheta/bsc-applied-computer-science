// process.cpp -- Jake Deery, 2018
#include "process.h"

int randGen(string fileToWrite, int n) {
	// local var: file handler
	ofstream fileHandler;

	// pen our file
	fileHandler.open(fileToWrite);

	// initiate the timer
	auto startTime = uTimeGet::now();

	// place random numbers until our power of 10 is met
	for (int itrtr = 0; itrtr < n; itrtr++) {
		fileHandler << rand() % 100 + 1 << " ";
	}

	// end our timer and calculate the value
	auto endTime = uTimeGet::now();
	deltaTime = endTime - startTime;

	// cleanup: close the file
	fileHandler.close();

	// function exit
	return 0;
}

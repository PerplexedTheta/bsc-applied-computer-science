// processes.cpp -- Jake Deery, 2018
#include "process.h"

// print a vector
/* -- DEFUNCT :(
int printVect(vector<int> srcVect) {
	for (int itrtr = 0; itrtr < srcVect.size(); itrtr++) {
		cout << srcVect[itrtr] << " "; // print out our values
	}

	return 0;
}
*/

// clear a vector, setting it to all 0s
int clearVect(vector<int> &srcVect) {
	// for loop: have we got to the end of the vector?
	for (int itrtr = 0; itrtr < srcVect.size(); itrtr++) {
		srcVect[itrtr] = 0; // set every value to 0
	}

	// kill func
	return 0;
}

// read from our txt file
int readFromFile(string fileToRead, vector<int> &destVect) {
	// local vars: handler and element
	fstream fileHandler(fileToRead, ios_base::in);
	if (fileHandler.fail()) {
		return -1;
	}

	double element;

	// clear the vect from any current values
	destVect.clear();

	// basic while loop: while last element hasnt been met (much like how cin works)
	while (fileHandler >> element) {
		// add it to our vector
		destVect.push_back((int)element);
	}

	// kill func
	return 0;
}

// write out to logfile
int writeLog(stringstream& toPrint, string fileToWrite) {
	// local var: file handler
	ofstream fileHandler;
	string str;

	// convert ss to str
	str = toPrint.str();

	// print our file
	fileHandler.open(fileToWrite, fstream::app);

	// print to our display
	cout << str;
	// print to our file
	fileHandler << str;

	// cleanup: close the file
	fileHandler.close();

	// function exit
	return 0;
}

// swap two integers within an int vector/array
int intSwap(int *int1, int *int2) {
	// temp store for swap
	int temp = *int1;

	// do our swap
	*int1 = *int2;
	*int2 = temp;

	// kill func
	return 0;
}

// bubblesort function
int bubbleSort(vector<int> &srcVect) {
	// initiate the timer
	auto startTime = uTimeGet::now();

	// for loop for detecting if we have reached the end of the vector
	for (int i = 0; i < srcVect.size() - 1; i++) {
		// for loop for finding out if we hit the 'j' limit, reducing by 1 per pass
		for (int j = 0; j < srcVect.size() - i - 1; j++) {
			// do our detection: swap values if true
			if (srcVect[j]> srcVect[j + 1]) {
				intSwap(&srcVect[j], &srcVect[j + 1]);
			}
		}
	}

	// end our timer and calculate the value
	auto endTime = uTimeGet::now();
	deltaTime = endTime - startTime;

	// exit func
	return 0;
}

// shellsort function
int shellSort(vector<int> &srcVect) {
	// initialise the timer
	auto startTime = uTimeGet::now();

	// start with a large gap, decreasing over time
	for (int gap = srcVect.size() / 2; gap > 0; gap /= 2) {
		// gap-sort our vector until we cant do that anymore
		for (int i = gap; i < srcVect.size(); i++) {
			// temporarily store element(i) in int
			int temp = srcVect[i];

			int j; // we store itrtr2 externally from this for loop because we need it later on
			for (j = i; j >= gap && srcVect[j - gap] > temp; j -= gap) {
				// shift up the gap-sorted elements until a suitable location for element(i) is found
				srcVect[j] = srcVect[j - gap];
			}

			// put element(i) back in its right place
			srcVect[j] = temp;
		}
	}

	// end our timer and calculate the value
	auto endTime = uTimeGet::now();
	deltaTime = endTime - startTime;

	// exit func
	return 0;
}

int insertionSort(vector<int> &srcVect) {
	// initialise the timer
	auto startTime = uTimeGet::now();

	int j, key;
	for (int i = 1; i < srcVect.size(); i++) {
		key = srcVect[i]; // store our current element in an int for later use
		j = i - 1; // make j point to the element behind i

		// while: j is more than 0, and the value of element(j) is more than element(i)
		while (j > -1 && srcVect[j] > key) {
			// swap j + 1 and j around
			intSwap(&srcVect[j + 1], &srcVect[j]);
			// then draw j back again and retry
			j = j - 1;
		}

		// once we escape that, move j+1 to position i
		srcVect[j + 1] = key;
	}

	// end our timer and calculate the value
	auto endTime = uTimeGet::now();
	deltaTime = endTime - startTime;

	// exit func
	return 0;
}

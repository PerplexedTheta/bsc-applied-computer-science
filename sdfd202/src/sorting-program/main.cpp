// SDFD202 -- Main/SortingProgram -- UF3366 CityPlym
// main.cpp -- Jake Deery, 2018
#include "main.h"

// vars for main_h
char keyPressed = 'h';
int tests = 0; // start at 10^1 = 10
int testsNext = 1; // start at 10^2 = 100
string fileName = "if-pow1.txt"; // start with 10^1 = 10
stringstream sstm;

int main() {
	// set our srand seed
	srand(time(NULL));

	// set up locale support
	#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		// no locale support
	#else
		cout.imbue(locale(""));
	#endif

	// welcome shit
	sstm << "Sort application v1 - Jake Deery, 2018" << endl << endl;
	writeLog(sstm, "of.txt");
	sstm.str(string());

	// check our text files -- 10
	cout << "Checking text file `if-pow1.txt` . . . ";
	// read in, hope we don't crash
	if(readFromFile("if-pow2.txt", randVect) == -1) {
		cout << "FILE FAILED!" << endl;
		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			system("pause");
		#else
			system("read -p 'Press ENTER to continue . . . '");
		#endif
		return 0;
	}
	// say done
	cout << "DONE" << endl;

	// check our text files -- 100
	cout << "Checking text file `if-pow2.txt` . . . ";
	if(readFromFile("if-pow2.txt", randVect) == -1) {
		cout << "FILE FAILED!" << endl;
		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			system("pause");
		#else
			system("read -p 'Press ENTER to continue . . . '");
		#endif
		return 0;
	}
	cout << "DONE" << endl;

	// check our text files -- 1.000
	cout << "Checking text file `if-pow3.txt` . . . ";
	if(readFromFile("if-pow2.txt", randVect) == -1) {
		cout << "FILE FAILED!" << endl;
		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			system("pause");
		#else
			system("read -p 'Press ENTER to continue . . . '");
		#endif
		return 0;
	}
	cout << "DONE" << endl;

	// check our text files -- 10.000
	cout << "Checking text file `if-pow4.txt` . . . ";
	if(readFromFile("if-pow2.txt", randVect) == -1) {
		cout << "FILE FAILED!" << endl;
		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			system("pause");
		#else
			system("read -p 'Press ENTER to continue . . . '");
		#endif
		return 0;
	}
	cout << "DONE" << endl << endl;

	// all done, wait for keypress
	callMenu(randVect);
	return 0;
}

// call up our menu
int callMenu(vector<int> srcVect) { // we pass our vectors for printing
	uTimeOut totalTime; // var for local time

	cout << "Press 1 to run a shell sort . . . " << endl;
	cout << "Press 2 to run a insertion sort . . . " << endl;
	cout << "Press 3 to run a bubble sort . . . " << endl;
	cout << "Press I to check " << pow(10, testsNext + 1) << " numbers . . . " << endl;
	cout << "Press Q to quit the application . . . " << endl;
	cout << "Press ENTER to accept choice or BkSpc to start again . . . " << endl;
	cout << "[1/2/3/I/Q]: ";
	cin >> keyPressed;
	cout << endl << endl;

	if (keyPressed == '1') {
		// say we're sorting
		sstm << "Sorting " << pow(10, tests + 1) <<" numbers with `shell sort` . . . ";
		writeLog(sstm, "of.txt");
		sstm.str(string());

		// clear the vector before we begin
		srcVect.clear();

		// do our sort 10x
		for (int runs = 0; runs < 10; runs++) {
			readFromFile(fileName, srcVect);
			shellSort(srcVect);
			totalTime += deltaTime;
		}

		// print the result
		sstm << "DONE in " << (totalTime.count() / 1000) / 10 << "microseconds (avg of 10 runs)" << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());

		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == '2') {
		sstm << "Sorting " << pow(10, tests + 1) << " numbers with `insertion sort` . . . ";
		writeLog(sstm, "of.txt");
		sstm.str(string());

		srcVect.clear();

		for (int runs = 0; runs < 10; runs++) {
			readFromFile(fileName, srcVect);
			insertionSort(srcVect);
			totalTime += deltaTime;
		}

		sstm << "DONE in " << (totalTime.count() / 1000) / 10 << "microseconds (avg of 10 runs)" << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());

		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == '3') {
		sstm << "Sorting " << pow(10, tests + 1) << " numbers with `bubble sort` . . . ";
		writeLog(sstm, "of.txt");
		sstm.str(string());

		srcVect.clear();

		for (int runs = 0; runs < 10; runs++) {
			readFromFile(fileName, srcVect);
			bubbleSort(srcVect);
			totalTime += deltaTime;
		}

		sstm << "DONE in " << (totalTime.count() / 1000) / 10 << "microseconds (avg of 10 runs)" << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());

		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == 'i' || keyPressed == 'I') { // both lower and uppercase values are accounted for
		if (tests == 0) {
			// increment both tests and testNext
			tests++;
			testsNext++;
			fileName = "if-pow2.txt";
			sstm.str(string());

			// tell us whats happening
			cout << "Will test " << pow(10, tests + 1) << " numbers using `" << fileName << "` . . . " << endl << endl;
		} else if (tests == 1) {
			tests++;
			testsNext++;
			fileName = "if-pow3.txt";

			// tell us whats happening
			cout << "Will test " << pow(10, tests + 1) << " numbers using `" << fileName << "` . . . " << endl << endl;
		} else if (tests == 2) {
			tests++;
			testsNext = 0; // say our next test is 10^1 = 10
			fileName = "if-pow4.txt";

			// tell us whats happening
			cout << "Will test " << pow(10, tests + 1) << " numbers using `" << fileName << "` . . . " << endl << endl;
			sstm.str(string());
		} else if (tests == 3) {
			tests = 0; // say our next value to test is 10^1 = 10 AND reset the statements
			testsNext++;
			fileName = "if-pow1.txt";

			// tell us whats happening
			cout << "Will test " << pow(10, tests + 1) << " numbers using `" << fileName << "` . . . " << endl << endl;
		}



	} else if (keyPressed == 'q' || keyPressed == 'Q') {



		// kill
		return 0;
	} else {



		// do nothing
	}

	// infinite loop
	callMenu(srcVect);
	return 0;
}

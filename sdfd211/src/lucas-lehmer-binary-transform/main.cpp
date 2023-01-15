// SDFD211 -- UF3366 CityPlym
// main.cpp -- Jake Deery, 2018
#include "main.h"

int main(int argc, char* argv[]) {
	// set up locale support
	#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		// no locale support
	#else
		cout.imbue(locale(""));
	#endif

	// welcome shit
	cout << "Lucas-Lehmer binary transform application v1" << endl;

	// ask the user for the prime number
	cout << "Enter prime number: ";
	cin >> prime;
	cout << endl << "Processing, please wait . . . ";

	// set up the vectors
	vector<int16_t> vect2(prime * 4, 0);
	vector<int16_t> vect1(prime * 4, 0);

	// make our vector squarable
	vect1[2] = 1;

	// initiate the timer
	auto startTime = uTimeGet::now();

	for (int16_t itrtr = 0; itrtr < prime - 2; itrtr++) {
		// Square vectorA, and store result in vectorB
		squareVect(vect1, vect2);

		// carry our vector
		carryVect(vect2);

		// take two from our vector
		minusTwo(vect2);

		// mod our vector
		modVect(vect2, vect1);

		// reset the vector again
		memset(&vect2[0], 0, sizeof(vect2[0]) * vect2.size());
	}

	// end our timer and calculate the value
	auto endTime = uTimeGet::now();
	uTimeOut deltaTime = endTime - startTime;

	cout << endl << "Final result = ";
	printVect(vect1);
	cout << endl;

	cout << "Delta time = " << deltaTime.count() / 1000000000 << " seconds" << endl;

	// all done, wait for keypress
	#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		system("pause");
	#else
		system("read -p 'Press ENTER to continue . . . '");
	#endif
	return 0;
}

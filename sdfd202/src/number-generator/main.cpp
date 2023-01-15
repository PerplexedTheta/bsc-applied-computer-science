// SDFD202 -- Tools/GenerateFile -- UF3366 CityPlym
// main.cpp -- Jake Deery, 2018
#include "main.h"

int main() {
	// set our srand seed
	srand(time(NULL));

	// set our locale so numbers display properly
	cout.imbue(locale(""));

	// welcome shit
	cout << "File generator application v1" << endl << endl;

	// all done, wait for keypress
	cout << endl;
	callMenu();
	return 0;
}

// call up our menu
int callMenu() {
	char keyPressed = '#';
	uint64_t powOf = 0;

	cout << "Press 1 to generate " << 10 << " number . . . " << endl;
	cout << "Press 2 to generate " << 100 << " number . . . " << endl;
	cout << "Press 3 to generate " << 1000 << " number . . . " << endl;
	cout << "Press 4 to generate " << 10000 << " number . . . " << endl;
	cout << "Press 5 to generate " << 100000 << " number . . . " << endl;
	cout << "Press 6 to generate " << 1000000 << " number . . . " << endl;
	cout << "Press 7 to generate " << 10000000 << " number . . . " << endl;
	cout << "Press 8 to generate " << 100000000 << " number . . . " << endl;
	cout << "Press 9 to generate " << 1000000000 << " number . . . " << endl;
	cout << "Press 0 to generate " << 1 << " number . . . " << endl;
	cout << "Press Q to exit this program . . . " << endl;
	cout << "Press ENTER to accept choice or BkSpc to start again . . . " << endl;
	cout << "[0-9/Q]: ";
	cin >> keyPressed;
	cout << endl;

	if (keyPressed == '0') {
		// set our values to calculate
		powOf = pow(10, 0);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '1') {
		// set our values to calculate
		powOf = pow(10, 1);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '2') {
		// set our values to calculate
		powOf = pow(10, 2);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '3') {
		// set our values to calculate
		powOf = pow(10, 3);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '4') {
		// set our values to calculate
		powOf = pow(10, 4);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '5') {
		// set our values to calculate
		powOf = pow(10, 5);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '6') {
		// set our values to calculate
		powOf = pow(10, 6);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '7') {
		// set our values to calculate
		powOf = pow(10, 7);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '8') {
		// set our values to calculate
		powOf = pow(10, 8);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == '9') {
		// set our values to calculate
		powOf = pow(10, 9);

		// say what we're doing
		cout << "Processing " << powOf << " number . . . ";

		// do the process
		randGen("of.txt", powOf);

		// tell the user we're done
		cout << "DONE in " << deltaTime.count() / 1000 << "ms" << endl;
		cout << endl;



	} else if (keyPressed == 'q' || keyPressed == 'Q') { // caps or lowercase Q
		// kill
		return 0;
	} else {
		// do nothing
	}

	// infinite loop
	callMenu();
	return 0;
}

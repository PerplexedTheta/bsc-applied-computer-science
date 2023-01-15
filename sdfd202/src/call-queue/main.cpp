// SDFD202 -- Main/CallQueue -- UF3366, CityPlym
// main.cpp, Jake Deery 2018
#include "main.h"

stringstream sstm;
int16_t idGen = 0;

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
	sstm << "Call queue application v1" << endl << endl;
	writeLog(sstm, "of.txt");
	sstm.str(string());

	// all done, wait for keypress
	callMenu(idVect, numVect, timeVect, avgCallsVect);
	return 0;



}

int16_t callMenu(vector<int16_t> idVect, vector<int64_t> numVect, vector<int32_t> timeVect, vector<int16_t> avgCallsVect) {
	char keyPressed = '#';
	int16_t deleteHere = 0;
	if (idVect.size() > 0) {
		deleteHere = rand() % idVect.size();
	}
	int16_t numSeed = rand() % 8192;

	cout << "Press 1 to add a call to the queue . . . " << endl; // add end
	cout << "Press 2 to put a call through to the operator . . . " << endl; // delete begin
	cout << "Press 3 to remove a call from the queue . . . " << endl; // delete rand
	cout << "Press 4 to update call times . . . " << endl; // reprint
	cout << "Press Q to quit the application . . . " << endl;
	cout << "Press ENTER to accept choice or BkSpc to start again . . . " << endl;
	cout << "[1/2/3/Q]: ";
	cin >> keyPressed;
	cout << endl << endl;

	if (keyPressed == '1') {
		if (idVect.size() < 20) {
			// add item to the end of the vector
			idGen++; // id number iterator
			idVect.push_back(idGen);
			numVect.push_back(7000000000 + (numSeed * 122070)); // 8192*122070 = 999997440, random telephone no
			timeVect.push_back(time(NULL));
			avgCallsVect.push_back(idVect.size()); // the average calls vector just stores a record of the idVect's size
			cout << "collapseVect: " << endl;
		} else {
			// if we exceed 20 calls, take no more
			sstm << "QUEUE FULL!" << endl;
			writeLog(sstm, "of.txt");
			sstm.str(string());
		}

		showAvg(timeVect, avgCallsVect, "of.txt");
		sstm << "Call queue: " << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());
		// print the vector result
		printVect(idVect, numVect, timeVect, "of.txt");
		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == '2') {
		if (idVect.size() == 0) {
			// dont try end any more calls if there are none to begin with
			sstm << "QUEUE EMPTY!" << endl;
			writeLog(sstm, "of.txt");
			sstm.str(string());
		} else {
			// delete item from begining of the vector
			idVect.erase(idVect.begin());
			numVect.erase(numVect.begin());
			timeVect.erase(timeVect.begin());
			avgCallsVect.push_back(idVect.size()); // we need to know when something is deleted, too
		}

		showAvg(timeVect, avgCallsVect, "of.txt");
		sstm << "Call queue: " << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());
		// print the vector result
		printVect(idVect, numVect, timeVect, "of.txt");
		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == '3') {
		// delete an item randomly from the vector
		if (idVect.size() == 0) {
			sstm << "QUEUE EMPTY!" << endl;
			writeLog(sstm, "of.txt");
			sstm.str(string());
		}
		else {
			idVect.erase(idVect.begin() + deleteHere); // deleteHere is randomly seeded every time we recall our menu
			numVect.erase(numVect.begin() + deleteHere);
			timeVect.erase(timeVect.begin() + deleteHere);
			avgCallsVect.push_back(idVect.size());
		}

		showAvg(timeVect, avgCallsVect, "of.txt");
		sstm << "Call queue: " << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());
		// print the vector result
		printVect(idVect, numVect, timeVect, "of.txt");
		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == '4') {
		// just reprint the queue
		showAvg(timeVect, avgCallsVect, "of.txt");
		sstm << "Call queue: " << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());
		// print the vector result
		printVect(idVect, numVect, timeVect, "of.txt");
		sstm << endl;
		writeLog(sstm, "of.txt");
		sstm.str(string());



	} else if (keyPressed == 'q' || keyPressed == 'Q') { // account for both cases of Q
		// kill
		return 0;



	} else {
		// do nothing



	}



	// infinite loop
	callMenu(idVect, numVect, timeVect, avgCallsVect);
	return 0;
}

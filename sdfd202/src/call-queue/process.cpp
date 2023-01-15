// global.cpp, Jake Deery 2018
#include "global.h"

// log to a file
int16_t writeLog(stringstream& toprint, string fileToWrite) {
	// local var: file handler
	ofstream fileHandler;
	string str;

	// convert ss to str
	str = toprint.str();

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

// print a vector
// SPECIALLY ADAPTED
int16_t printVect(vector<int16_t> idVect, vector<int64_t> numVect, vector<int32_t> timeVect, string fileName) {
	// this lets us write to log and to screen
	stringstream sstm;

	for (int16_t i = 0; i < idVect.size(); i++) {
		// delta time is the difference between now and then
		deltaTime = time(NULL) - timeVect[i];

		// print our magic
		sstm << "(" << idVect[i] << ") 0" << numVect[i] << " -- Active for " << deltaTime << " seconds . . . " << endl;
		writeLog(sstm, fileName);
		sstm.str(string());
	}

	return 0;
}

// clear a vector, setting it to all 0s
int16_t clearVect(vector<int64_t> &srcVect) {
	// for loop: have we got to the end of the vector?
	for (int16_t i = 0; i < srcVect.size(); i++) {
		srcVect[i] = 0; // set every value to 0
	}

	// kill func
	return 0;
}

// show the average time passed
int16_t showAvg(vector<int32_t> timeVect, vector<int16_t> avgCallsVect, string fileName) {
	vector<int64_t> tempVect;
	int64_t avgWaitTime = 0;
	float avgCallsNo = 0;
	stringstream sstm;

	if (timeVect.size() > 0) {
		// first off, show the user the average wait time
		for (int16_t i = 0; i < timeVect.size(); i++) { // push the number of seconds passed to the avgWaitTime int64_t
			tempVect.push_back(time(NULL) - timeVect[i]);
			avgWaitTime += tempVect[i];
		}

		// next, calculate the average number of active calls
		for (int16_t i = 0; i < avgCallsVect.size(); i++) { // push the number of seconds passed to the avgWaitTime int64_t
			avgCallsNo += avgCallsVect[i];
		}

		sstm << "Average wait time: " << avgWaitTime / timeVect.size() << endl; // divide the avgWaitTime by the vect size to get an average
		writeLog(sstm, fileName);
		sstm.str(string());

		sstm << "Average number of calls: " << avgCallsNo / avgCallsVect.size() << endl; // divide the avgCallsNo by the vect size to get an average
		writeLog(sstm, fileName);
		sstm.str(string());
	} else {
		// do nothing
	}

	return 0;
};

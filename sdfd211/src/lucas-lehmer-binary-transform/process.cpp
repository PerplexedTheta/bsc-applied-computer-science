// processes.cpp -- Jake Deery, 2018
#include "process.h"

// print a vector
int16_t printVect(vector<int16_t> srcVect) {
	for (int16_t i = srcVect.size() - 1; i > -1; i--) {
		cout << srcVect[i]; // print out our values BACKWARDS (binary reads right-to-left)
	}

	return 0;
}

/*int16_t clearVect(vector<int16_t> &destVect) { // dead function -- use memset (faster)
	for (int16_t i = 0; i < destVect.size(); i++) {
		destVect[i] = 0;
	}
}*/

// square a vector
int16_t squareVect(vector<int16_t> srcVect, vector<int16_t> &destVect) {
	// clear destVect
	memset(&destVect[0], 0, sizeof(destVect[0]) * destVect.size());

	// do the maths
	for (int16_t i = 0; i < srcVect.size() - 1; i++) {
		if (srcVect[i] == 1) {
			for (int16_t j = 0; j < srcVect.size() - 1; j++) {
				if (srcVect[j] == 1) {
					destVect[i + j]++;
				}
			}
		}
	}

	return 0;
}

int16_t carryVect(vector <int16_t> &srcVect) {
	for (int16_t i = 0; i < srcVect.size(); i++) {
		// check our binary shifts starting at the largest number
		if (srcVect[i] > 16383) {
			srcVect[i] -= 16384;
			srcVect[i + 14]++;
		}

		if (srcVect[i] > 8191) {
			srcVect[i] -= 8192;
			srcVect[i + 13]++;
		}

		if (srcVect[i] > 4095) {
			srcVect[i] -= 4096;
			srcVect[i + 12]++;
		}

		if (srcVect[i] > 2047) {
			srcVect[i] -= 2048;
			srcVect[i + 11]++;
		}

		if (srcVect[i] > 1023) {
			srcVect[i] -= 1024;
			srcVect[i + 10]++;
		}

		if (srcVect[i] > 511) {
			srcVect[i] -= 512;
			srcVect[i + 9]++;
		}

		if (srcVect[i] > 255) {
			srcVect[i] -= 256;
			srcVect[i + 8]++;
		}

		if (srcVect[i] > 127) {
			srcVect[i] -= 128;
			srcVect[i + 7]++;
		}

		if (srcVect[i] > 63) {
			srcVect[i] -= 64;
			srcVect[i + 6]++;
		}

		if (srcVect[i] > 31) {
			srcVect[i] -= 32;
			srcVect[i + 5]++;
		}

		if (srcVect[i] > 15) {
			srcVect[i] -= 16;
			srcVect[i + 4]++;
		}

		if (srcVect[i] > 7) {
			srcVect[i] -= 8;
			srcVect[i + 3]++;
		}

		if (srcVect[i] > 3) {
			srcVect[i] -= 4;
			srcVect[i + 2]++;
		}

		if (srcVect[i] > 1) {
			srcVect[i] -= 2;
			srcVect[i + 1]++;
		}
	}

	return 0;
}

// take two from a binary vector
int16_t minusTwo(vector <int16_t> &srcVect) {
	int16_t z;

	// Check if the 2s unit == 1. If so, change to zero
	if (srcVect[1] == 1) {
		srcVect[1] = 0;
	} else {
		for (int16_t i = 2; i < srcVect.size(); i++) {
			if (srcVect[i] == 1) {
				z = i;
				srcVect[z] = 0;
				i = srcVect.size();
			}
		}

		// Change preceeding values from 0 to 1
		for (int16_t i = z - 1; i > 0; i--) {
			srcVect[i] = 1;
		}
	}

	return 0;
}

// mod a vector!
int16_t modVect(vector<int16_t> srcVect, vector<int16_t> &destVect) {
	int16_t k = 0;

	// clear destVect
	memset(&destVect[0], 0, sizeof(destVect[0]) * destVect.size());

	// Copy P digits from srcVect to destVect
	while (k < srcVect.size()) {
		for (int16_t i = 0; i < prime; i++) {
			destVect[i] += srcVect[i + k];

			if (destVect[i] > 1) {
				destVect[i] -= 2;
				destVect[i + 1]++;
			}
		}

		k += prime;

	}

	return 0;
}

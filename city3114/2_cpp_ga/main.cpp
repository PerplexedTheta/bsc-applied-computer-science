// main.cpp, city3114_proc2, jake deery, 2020
#include "main.h"
#include "gaClass.h"

int main(int argc, char** argv) {
	// configure srand
	srand(time(NULL));
	
	// configure locale
	locale systemLocale("fr_FR.UTF-8");
	cout.imbue(systemLocale);
	
	// say hi
	cout << "[I] CITY3114_PROC2, Jake Deery, 2020" << "\n";
	
	// create the object
	gaClass* gaClassObj = new gaClass();
	
	// read the class cpp to see what's up
	gaClassObj->runProc();
	
	// delete object
	delete gaClassObj;
	
	return 0;
}

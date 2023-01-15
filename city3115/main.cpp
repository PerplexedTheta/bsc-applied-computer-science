// city3115, main.cpp, jake deery, 2020
#include "main.h"

int main(int argc, char* argv[]) {
	// vars
	int retVal = -1;
	VideoCapture capWebcam(0);

	// into
	cout << "[I] taskProgram, Jake Deery, 2020 . . . " << "\n";
	cout << "[W] OpenCV is system resource intensive, please use at your own risk . . . " << "\n";
	
	// create new object
	opticalFlow * opticalFlowObj = new opticalFlow(capWebcam);
	
	// do the lucas-kanade process
	retVal = opticalFlowObj->doSparseProcess();
	
	// do the farneback process
	retVal = opticalFlowObj->doDenseProcess();

	// end program
	delete opticalFlowObj;
	cout << "[I] Class returned code " << retVal << " . . . "  << "\n";
	
	cout << "[E] Program closed with code " << retVal << " . . . " << "\n";
	return retVal;
}

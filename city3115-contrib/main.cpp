// taskProgram, main.cpp, jake deery, 2020
#include "main.h"

int main(int argc, char* argv[]) {
	// vars
	int retVal = -1;

	// into
	cout << "[I] taskProgram, Jake Deery, 2020 . . . " << "\n";
	cout << "[W] OpenCV is system resource intensive, please use at your own risk . . . " << "\n";
	
	// comment out unwanted methods
	retVal = doCvtColour();
	retVal = doEdgeDetection();
	retVal = doGaussianMixtureModel();
	retVal = doOpticalFlow();
	
	cout << "[E] Program closed with code " << retVal << " . . . " << "\n";
	return retVal;
}

int doCvtColour() {
	// create new object
	int retVal = -1;
	cvtColour * cvtColourObj = new cvtColour("lenna.png");
	
	// call upon main function
	retVal = cvtColourObj->doImageProcess("bgr2rgb", COLOR_BGR2RGB);
	retVal = cvtColourObj->doImageProcess("bgr2grey", COLOR_BGR2GRAY);
	
	retVal = cvtColourObj->doImageProcess("rgb2ycrcb", COLOR_RGB2YCrCb);
	retVal = cvtColourObj->doImageProcess("bgr2ycrcb", COLOR_BGR2YCrCb);
	
	retVal = cvtColourObj->doImageProcess("rgb2hsv", COLOR_RGB2HSV);
	retVal = cvtColourObj->doImageProcess("bgr2hsv", COLOR_BGR2HSV);
	
	retVal = cvtColourObj->doImageProcess("rgb2lab", COLOR_RGB2Lab);
	retVal = cvtColourObj->doImageProcess("bgr2lab", COLOR_BGR2Lab);
	
	retVal = cvtColourObj->doImageProcess("rgb2luv", COLOR_RGB2Luv);
	retVal = cvtColourObj->doImageProcess("bgr2luv", COLOR_BGR2Luv);
	
	retVal = cvtColourObj->doImageProcess("rgb2xyz", COLOR_RGB2XYZ);
	retVal = cvtColourObj->doImageProcess("bgr2xyz", COLOR_BGR2XYZ);
	
	// end program
	waitKey(0);
	delete cvtColourObj;
	cout << "[I] Class returned code " << retVal << " . . . " << "\n";
	return retVal;
}

int doEdgeDetection() {
	// vars
	int retVal = -1;
	VideoCapture capWebcam(0);
	
	// create new object
	edgeDetection * edgeDetectionObj = new edgeDetection(capWebcam);

	// do our process
	retVal = edgeDetectionObj->doEdgeDetection();

	// end program
	delete edgeDetectionObj;
	cout << "[I] Class returned code " << retVal << " . . . "  << "\n";
	return retVal;
}

int doGaussianMixtureModel() {
	// vars
	int retVal = -1;
	VideoCapture capWebcam(0);

	// create new object
	gaussianMixtureModels * gaussianMixtureModelsObj = new gaussianMixtureModels(capWebcam);
	
	// do the process
	retVal = gaussianMixtureModelsObj->doMotionDetection(createBackgroundSubtractorMOG2());

	// end program
	delete gaussianMixtureModelsObj;
	cout << "[I] Class returned code " << retVal << " . . . "  << "\n";
	return retVal;
}

int doOpticalFlow() {
	// vars
	int retVal = -1;
	VideoCapture capWebcam(0);
	
	// create new object
	opticalFlow * opticalFlowObj = new opticalFlow(capWebcam);
	
	// do the first process
	retVal = opticalFlowObj->doDenseProcess();

	// do the second process
	retVal = opticalFlowObj->doSparseProcess();

	// end program
	delete opticalFlowObj;
	cout << "[I] Class returned code " << retVal << " . . . "  << "\n";
	return retVal;
}

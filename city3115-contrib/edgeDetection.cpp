// edgeDetection.cpp, jake deery, 2020
#include "edgeDetection.h"

edgeDetection::edgeDetection(VideoCapture inputVideo) {
	// init - load vars into object
	capSource = inputVideo;
	
	// Check for failure
	if(capSource.isOpened() == false) {
		cout << "[E] Could not open or find the webcam . . . " << "\n";
		delete this;
	}
	
	// recalculate the fps value
	if((fps / 1000) < 1) fps = 1;
	else fps = ceil(fps / 1000);
	
	cout << "[I] Class created successfully . . . " << "\n";
}

edgeDetection::~edgeDetection() {
	// destructor - delete windows
	cout << "[I] Deleting all windows . . . " << "\n";
	destroyAllWindows();
	
	cout << "[I] Class deleted successfully . . . " << "\n";
}

// define DRY method
Mat edgeDetection::cannyThreshold(Mat src, int gausKSize, double sigmaX, double lowThreshold, double ratio,  int cannyKSize) {
	// vars
	Mat temp;
	Mat dest;

	// blur the image to make it easier to find the edges, and then do the canny process
	GaussianBlur(src, temp, Size(gausKSize, gausKSize), sigmaX); // gaussian curve
	Canny(temp, dest, lowThreshold, lowThreshold * ratio, cannyKSize); // 

	// return the material
	return dest;
}

int edgeDetection::doEdgeDetection() {
	// vars
	String windowName = "doEdgeDetection";
	Mat srcMat;
	Mat destMat;
	char checkForEscKey;

	// intro
	cout << "[I] Calling on method doEdgeDetection . . . " << "\n";
	
	// main webcam loop
	cout << "[W] Entering program loop . . . " << "\n";
	while(checkForEscKey != 27) {
		// read image to src from webcam and check for errors
		capSource >> srcMat;
		if(srcMat.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
		}
		
		// flip the frame for natural movement
		flip(srcMat, srcMat, 1);
		
		// do the edge detection process
		destMat = cannyThreshold(srcMat, gausKSize, sigmaX, lowThreshold, ratio, cannyKSize);

		// render
		namedWindow(windowName);
		imshow(windowName, destMat);
		
		// detect a break
		checkForEscKey = waitKey(fps);
	}
	
	return 0;
}

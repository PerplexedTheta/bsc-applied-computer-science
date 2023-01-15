// gaussianMixtureModels.cpp, jake deery, 2020
#include "gaussianMixtureModels.h"

gaussianMixtureModels::gaussianMixtureModels(VideoCapture inputVideo) {
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

gaussianMixtureModels::~gaussianMixtureModels() {
	// destructor - delete windows
	cout << "[I] Deleting all windows . . . " << "\n";
	destroyAllWindows();
	
	cout << "[I] Class deleted successfully . . . " << "\n";
}

int gaussianMixtureModels::doMotionDetection(Ptr<BackgroundSubtractor> inputBackSub) {
	// init
	pBackSub = inputBackSub;

	// vars
	Mat frame;
	Mat mask;
	stringstream ss;
	char checkForEscKey;
	
	// intro
	cout << "[I] Calling on method doMotionDetection . . . " << "\n";

	// begin process
	cout << "[W] Entering program loop . . . " << "\n";
	while (checkForEscKey != 27) {
		// grab a frame
		capSource >> frame;
		if (frame.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
		}

		// get the material ready for processing
		flip(frame, frame, 1);

		//update the background model
		pBackSub->apply(frame, mask);

		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);

		ss << capSource.get(CAP_PROP_POS_FRAMES);

		String frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("Mask", mask);

		// detect exit
		checkForEscKey = waitKey(fps);
	}
	return 0;
}
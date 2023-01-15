// opticalFlow.cpp, jake deery, 2020
#include "opticalFlow.h"

opticalFlow::opticalFlow(VideoCapture inputVideo) {
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

opticalFlow::~opticalFlow() {
	// destructor - delete windows
	cout << "[I] Deleting all windows . . . " << "\n";
	destroyAllWindows();
	
	cout << "[I] Class deleted successfully . . . " << "\n";
}

int opticalFlow::doDenseProcess() {
	// vars
	Mat frame1;
	Mat prvs;
	char checkForEscKey;

	// intro
	cout << "[I] Calling on method doDenseProcess . . . " << "\n";

	// copy webcam frame to Mat & make it grey
	capSource >> frame1;
	if (frame1.empty()) {
		cout << "[E] Could not open or find the webcam . . . " << "\n";
		return -1;
	}
	// get the material ready for processing
	flip(frame1, frame1, 1);
	cvtColor(frame1, prvs, COLOR_BGR2GRAY);
	
	// create blank window
	namedWindow("doDenseProcess");
	
	// begin process
	cout << "[W] Entering program loop . . . " << "\n";
	while (checkForEscKey != 27) {
		// vars
		Mat frame2;
		Mat next;
		Mat flow_parts[2];
		Mat magnitude;
		Mat angle;
		Mat magn_norm;
		Mat flow(prvs.size(), CV_32FC2);
		Mat _hsv[3];
		Mat hsv;
		Mat hsv8;
		Mat bgr;

		// copy webcam frame to Mat & make it grey
		capSource >> frame2;
		if (frame2.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
			break;
		}
		// get the material ready for processing
		flip(frame2, frame2, 1);
		cvtColor(frame2, next, COLOR_BGR2GRAY);

		// calculate the flow
		calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		// visualise the flow
		split(flow, flow_parts);
		cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
		angle *= ((1.f / 360.f) * (180.f / 255.f));

		//build hsv image
		_hsv[0] = angle;
		_hsv[1] = Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);
		cvtColor(hsv8, bgr, COLOR_HSV2BGR);

		// display the image
		imshow("doDenseProcess", bgr);

		// detect exit
		checkForEscKey = waitKey(fps);

		// blit
		prvs = next;
	}

	return 0;
}

int opticalFlow::doSparseProcess() {
	// vars
	Mat oldFrame;
	Mat oldGrey;
	Mat mask;
	RNG rng;
	vector<Scalar> colors;
	vector<Point2f> p0;
	vector<Point2f> p1;
	char checkForEscKey;

	// intro
	cout << "[I] Calling on method doSparseProcess . . . " << "\n";

	// create some random colours
	for(int i = 0; i < 100; i++) {
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r,g,b));
	}

	// take first frame
	capSource >> oldFrame;
	if (oldFrame.empty()) {
		cout << "[E] Could not open or find the webcam . . . " << "\n";
		return -1;
	}
	
	// flip the frame for natural movement
	flip(oldFrame, oldFrame, 1);

	// find corners in the mat
	cvtColor(oldFrame, oldGrey, COLOR_BGR2GRAY);
	goodFeaturesToTrack(oldGrey, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

	// create a mask image for drawing purposes
	mask = Mat::zeros(oldFrame.size(), oldFrame.type());
	
	// create blank window
	namedWindow("doSparseProcess");

	cout << "[W] Entering program loop . . . " << "\n";
	while(checkForEscKey != 27) {
		// vars
		Mat frame;
		Mat frameGrey;
		Mat img;
		vector<Point2f> goodNew;
		vector<uchar> status;
		vector<float> err;

		// copy frame to mat
		capSource >> frame;
		if (frame.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
			break;
		}
		// flip the frame for natural movement
		flip(frame, frame, 1);
		
		// prep the mat
		cvtColor(frame, frameGrey, COLOR_BGR2GRAY);

		// do the special stuff (optical flow)
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(oldGrey, frameGrey, p0, p1, status, err, Size(15,15), 2, criteria);

		for(uint i = 0; i < p0.size(); i++) {
			// select good points
			if(status[i] == 1) {
				goodNew.push_back(p1[i]);
				// draw the tracks
				line(mask,p1[i], p0[i], colors[i], 2);
				circle(frame, p1[i], 5, colors[i], -1);
			}
		}

		add(frame, mask, img);
		imshow("doSparseProcess", img);
	
		// detect exit
		checkForEscKey = waitKey(fps);

		// now update the previous frame and previous points
		oldGrey = frameGrey.clone();
		p0 = goodNew;
   }

	return 0;
}

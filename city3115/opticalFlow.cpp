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
	else fps = ceil(fps / 1000); // some accuracy issues here but its fine
	
	cout << "[I] Class created successfully . . . " << "\n";
}

opticalFlow::~opticalFlow() {
	// destructor - delete windows
	cout << "[I] Deleting all windows . . . " << "\n";
	destroyAllWindows();
	
	cout << "[I] Class deleted successfully . . . " << "\n";
}

// do luscan-kanade process
int opticalFlow::doSparseProcess() {
	// vars
	Mat prevFrame; // this var and the next refer to previous frames to compare
	Mat prevGrey;
	Mat mask; // this is a temporary var for us to draw dots on
	RNG rng; // this is a rng which we use for creating different colours
	vector<Scalar> colors; // here is a vector for keeping some random colours
	vector<Point2f> prevPts; // these vars are where the point values are kept
	vector<Point2f> nextPts; // for our model
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
	capSource >> prevFrame;
	if (prevFrame.empty()) {
		cout << "[E] Could not open or find the webcam . . . " << "\n";
		return -1;
	}
	
	// get the material ready for processing (greyscale images reduce problem size
	// by roughly 2/3rds) and flip it (i.e. non-mirrored)
	flip(prevFrame, prevFrame, 1);
	cvtColor(prevFrame, prevGrey, COLOR_BGR2GRAY);

	// find corners in the mat
	// flag 1: image to test
	// flag 2: the vector to store the found corners in
	// flag 3: the maximum allowed number of points to track
	// flag 4: minimum 'quality level' specified by eigenval algorithm
	// note: eigenval is generally considered better for faces
	//
	// flag 5: the minimum distance between each corner
	// flag 6: optional RoI mask or (in my case) empty
	// flag 7: block size for covariance matrix
	// flag 8: false for eigenval algo, true for harris detector algo
	// flag 9: k-value, only valid for harris detector
	goodFeaturesToTrack(
						prevGrey, // 1
						prevPts, //  2
						100, //      3
						0.3, //      4
						7, //        5
						Mat(), //    6
						7, //        7
						false, //    8
						0 //         9
	);

	// create a mask image for drawing purposes
	mask = Mat::zeros(prevFrame.size(), prevFrame.type());
	
	// create blank window
	namedWindow("doSparseProcess");

	cout << "[W] Entering program loop . . . " << "\n";
	while(checkForEscKey != 27) {
		// vars
		Mat nextFrame; // as above, these 'nextFrame'/current vars are temporary
		Mat nextGrey; //  and are only required for this loop to compare against old
		Mat img; // this will hopefully be our rendered image
		vector<Point2f> goodNew;
		vector<uchar> status;
		vector<float> error;
		TermCriteria criteria; // termination critera for the algorithm

		// copy frame to mat
		capSource >> nextFrame; // copy a new frame into the var
		if (nextFrame.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
			break;
		}
		// flip the frame for natural movement and prep the mat as before
		flip(nextFrame, nextFrame, 1);
		cvtColor(nextFrame, nextGrey, COLOR_BGR2GRAY);

		// calculate termination criteria, using flag 3 as our epsilon
		criteria = TermCriteria((TermCriteria::COUNT + TermCriteria::EPS), 10, 0.03);
		
		// do the special stuff (lucas-kanade)
		// flag 1: initial input frame to compare against
		// flag 2: the next frame to compare against
		// flag 3: the vector of points where the last iteration found
		// flag 4: the output vector where the algo will output its prediction to
		// flag 5: vector of status codes, 1 for flow found and 0 for flow loss
		// flag 6: vector of error codes, unused in my code but required by method
		// flag 7: window size used by each pyramid, used to increase robustness
		// at the expense of computational power -- smaller is less robust, 
		// just right is just right, larger is blurrier and expensive
		//
		// flag 8: pyramidal level number where 0 is disabled, 1 is 2, 2 is 3, etc
		// smaller is weaker, larger is more robust at the expense of performance
		//
		// flag 9: termination criteria, as specified above
		calcOpticalFlowPyrLK(
						prevGrey, //         1
						nextGrey, //         2
						prevPts, //          3
						nextPts, //          4
						status, //           5
						error, //            6
						Size(70,70), // flag 7: change to balance performance/robustness
						5, //           flag 8: change to balance performance/robustness
						criteria //          9
		);

		// here, we iterate through all previous points
		for(uint i = 0; i < prevPts.size(); i++) {
			// select good points found above, to speed up computation
			if(status[i] == 1) {
				// keep the good values from the current cycle
				goodNew.push_back(nextPts[i]);
				// draw the tracks onto the mask
				line(mask,nextPts[i], prevPts[i], colors[i], 2);
				circle(nextFrame, nextPts[i], 5, colors[i], -1);
			}
		}

		// blend the frame with the mask and render
		add(nextFrame, mask, img);
		imshow("doSparseProcess", img);
	
		// detect exit
		checkForEscKey = waitKey(fps);

		// now update the previous frame and previous points with the new values
		prevGrey = nextGrey.clone();
		prevPts = goodNew;
   }

	return 0;
}

// do farneback process
int opticalFlow::doDenseProcess() {
	// vars
	Mat prvs; // previous frame to compare, always in greyscale
	char checkForEscKey;

	// intro
	cout << "[I] Calling on method doDenseProcess . . . " << "\n";

	// copy webcam frame to Mat
	capSource >> prvs;
	if (prvs.empty()) {
		cout << "[E] Could not open or find the webcam . . . " << "\n";
		return -1;
	}
	// get the material ready for processing (greyscale images reduce problem size
	// by roughly 2/3rds) and flip it (i.e. non-mirrored)
	flip(prvs, prvs, 1);
	cvtColor(prvs, prvs, COLOR_BGR2GRAY);
	
	// create blank window
	namedWindow("doDenseProcess");
	
	// begin process
	cout << "[W] Entering program loop . . . " << "\n";
	while (checkForEscKey != 27) {
		// vars
		Mat next; // next frame to compare, again, always in greyscale
		Mat flow(prvs.size(), CV_32FC2);
		Mat flowParts[2]; // for splitting flow into x/y
		Mat angle; // for calculating angle hue
		Mat magnitude; // for calculating magnitude value (lightness)
		Mat _hsv[3]; // hsv array for image building
		Mat hsv; // for built image
		Mat rgb; // for rgb display image

		// copy webcam frame to Mat & make it grey
		capSource >> next;
		if (next.empty()) {
			cout << "[E] Could not open or find the webcam . . . " << "\n";
			return -1;
			break;
		}
		// get the material ready for processing
		flip(next, next, 1);
		cvtColor(next, next, COLOR_BGR2GRAY);

		// calculate the flow
		// flag 1: initial input frame to compare against
		// flag 2: "  " next input frame
		// flag 3: blank flow image with same size and colour profile as prev for
		// outputting the calculated flow pattern
		//
		// flag 4: pyramid scale, where 0,5 == standard pyramid (I see no compelling
		// argument to change this value)
		//
		// flag 5: pyramidal levels, where 1 is no extra layers, 2 is 1 extra, etc
		// flag 6: window size used by each pyramid, used to increase robustness
		// at the expense of computational power -- smaller is less robust, 
		// just right is just right, larger is blurrier and expensive
		//
		// flag 7: no of iterations per pyramidal level, more is always better at
		// the expense of computational performance
		//
		// flag 8: polynomial pixel neighbourhood size, where higher values offer
		// smoother but more blurred results at the cost of computational
		// performance
		//
		// flag 9: polynomial sigma (i.e. std deviation applied to the gaussian), 
		// which smooths derivatives used by the polynomial expansion process
		// note: value is closely tied to flag 8 and must be optimised together
		//
		// flag 10: flags, unused in my code
		calcOpticalFlowFarneback(
						prvs,  // 1
						next, //  2
						flow, //  3
						0.5, //   4
						2, //     5
						7, //    6
						2, //     7
						7, //     8
						1.5, //   9
						0 //     10
		);

		// split the flow into a 2D array
		split(flow, flowParts); 
		
		// calculate the angle and magnitude of the flow (i.e. the direction
		// and speed)
		//
		// flag 1: the input x-pos of the flow
		// flag 2: "  " y-pos
		// flag 3: the output magnitude image
		// flag 4: "  " angle image
		// flag 5: specify the calculations be done in deg (true) or rad (false)
		cartToPolar(
						flowParts[0], // 1
						flowParts[1], // 2
						magnitude, //    3
						angle, //        4
						true //          5
		);
		
		// calculate the angle hue colouring and normalise the magnitude for value
		angle *= ((1.f / 360.f) * (180.f / 255.f));
		normalize(magnitude, magnitude, 0.0f, 1.0f, NORM_MINMAX);

		//build hsv image -- copy calculations to array and merge
		_hsv[0] = angle;
		_hsv[1] = Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magnitude;
		merge(_hsv, 3, hsv);
		
		// make hsv mat a recognised colour space and convert it to rgb
		hsv.convertTo(hsv, CV_8U, 255.0);
		cvtColor(hsv, rgb, COLOR_HSV2RGB);

		// display the image
		imshow("doDenseProcess", rgb);

		// detect exit
		checkForEscKey = waitKey(1);

		// blit
		prvs = next;
	}

	return 0;
}

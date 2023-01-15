// opticalFlow.cpp, jake deery, 2020
#ifndef GAUSSIANMIXTUREMODELS_H
#define GAUSSIANMIXTUREMODELS_H

// include
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

// namespace
using namespace cv;
using std::cout;
using std::stringstream;

class gaussianMixtureModels {
public:
	// callable functions
	gaussianMixtureModels(VideoCapture inputVideo);
	~gaussianMixtureModels();

	// callable vars
	int doMotionDetection(Ptr<BackgroundSubtractor> inputBackSub);
private:
	// internal functions

	// internal vars
	VideoCapture capSource;
	Ptr<BackgroundSubtractor> pBackSub;
	int fps = 25;
};

#endif /* GAUSSIANMIXTUREMODELS_H */


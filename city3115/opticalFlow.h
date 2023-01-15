// opticalFlow.h, jake deery, 2020
#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

// include
#include <iostream>
#include <opencv2/opencv.hpp>

// namespace
using namespace cv;
using std::cout;
using std::vector;

class opticalFlow {
public:
	// callable functions
	opticalFlow(VideoCapture inputVideo);
	~opticalFlow();
	int doSparseProcess();
	int doDenseProcess();

	// callable vars

private:
	// internal functions

	// internal vars
	VideoCapture capSource;
	int fps = 25;
};

#endif /* OPTICALFLOW_H */

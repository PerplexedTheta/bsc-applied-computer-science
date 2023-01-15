// edgeDetection.cpp, jake deery, 2020
#ifndef EDGEDETECTION_H
#define EDGEDETECTION_H

// include
#include <iostream>
#include <opencv2/opencv.hpp>

// namespace
using namespace cv;
using std::cout;

class edgeDetection {
public:
	// callable functions
	edgeDetection(VideoCapture inputVideo);
	~edgeDetection();
	int doEdgeDetection();

	// callable vars
private:
	// internal functions
	Mat cannyThreshold(Mat src, int gausKSize, double sigmaX, double lowThreshold, double ratio,  int cannyKSize);

	// internal vars
	VideoCapture capSource;
	int fps = 25;
	const double sigmaX = 1.41;
	const double lowThreshold = 25.0;
	const double ratio = 3.0;
	const int gausKSize = 7;
	const int cannyKSize = 3;
};

#endif /* EDGEDETECTION_H */


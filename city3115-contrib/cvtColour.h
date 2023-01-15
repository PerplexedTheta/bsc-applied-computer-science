// cvtColour.h, jake deery, 2020
#ifndef CVTCOLOUR_H
#define CVTCOLOUR_H

// include
#include <iostream>
#include <opencv2/opencv.hpp>

// namespace
using namespace cv;
using std::cout;

class cvtColour {
public:
	// callable functions
	cvtColour(String inputImage);
	~cvtColour();
	int doImageProcess(String windowName, int processType);
	
	// callable vars
private:
	// internal functions
	
	// internal vars
	Mat loadedImage;
	Mat loadedImageGreyscale;
};

#endif /* CVTCOLOUR_H */

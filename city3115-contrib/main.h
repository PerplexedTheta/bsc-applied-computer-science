// taskProgram, main.h, jake deery, 2020
#ifndef MAIN_H
#define MAIN_H

// include
#include <iostream>

// class include for main
#include "cvtColour.h"
#include "edgeDetection.h"
#include "gaussianMixtureModels.h"
#include "opticalFlow.h"

// namespace
using std::cout;

// define functions
extern int doCvtColour();
extern int doEdgeDetection();
extern int doGaussianMixtureModel();
extern int doOpticalFlow();

#endif /* MAIN_H */

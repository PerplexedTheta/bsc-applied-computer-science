// cvtColour.cpp, jake deery, 2020
#include "cvtColour.h"

cvtColour::cvtColour(String inputImage) {
	// init - load vars into object
	loadedImage = imread(inputImage); // read in colour
	loadedImageGreyscale = imread(inputImage, IMREAD_GRAYSCALE); // read in greyscale
	
	// check image actually loaded
	if (loadedImage.empty()) {
		cout << "[E] Could not open or find " << inputImage << "\n";
		delete this; // kill object
	}
	
	// display the loaded images - colour
	cout << "[I] Displaying loaded image in colour . . . " << "\n";
	imshow("rgb", loadedImage);
	
	// display the loaded images - greyscale
	cout << "[I] Displaying loaded image in greyscale . . . " << "\n";
	imshow("greyscale", loadedImageGreyscale);
	imwrite("greyscale.png", loadedImageGreyscale);
	
	cout << "[I] Class created successfully . . . " << "\n";
}

cvtColour::~cvtColour() {
	// destructor - delete windows
	cout << "[I] Deleting all windows . . . " << "\n";
	destroyAllWindows();
	
	cout << "[I] Class deleted successfully . . . " << "\n";
}

int cvtColour::doImageProcess(String windowName, int processType) {
	// vars
	Mat outputImage;
	
	// intro
	cout << "[I] Calling on method doImageProcess . . . " << "\n";

	// process
	cvtColor(loadedImage, outputImage, processType);

	// display result
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, outputImage);
	imwrite(windowName.append(".png"), outputImage);
	
	// end process
	return 0;
}

// SDFD209 -- engine.cpp -- Jake Deery, 2018
#include "engine.h"

typedef GLubyte pixel[3]; // rgb representation
uint32_t texWidth, texHeight; // output file vars
pixel *image; // a new image to load
uint32_t frameCount, deltaTime, startTime, finishTime; // clocks for a timer
int tempClock = 0;
GLdouble rotX, rotY, rotZ;
double zoomFactor = 1.0; // camera modifier

Object3d* obj;

int allowedSize(int n) { // returns max ^2 <= n*
	int result;
	result = 1;

	while (result < n) {
		result = (result << 1); // bit shifting (<< or >>) is equal to multiplying or dividing by power of two, except its fast!
	}

	if (result == n) {
		return result;
	} else {
		return result >> 1;
	}
}

void readImage(const char *filename) { // readImage loads our image into memory for opengl to use
	int width, height, maxIntensity;
	unsigned int red, green, blue;
	int k;
	char content;
	FILE *texPointer;

	// firstly, lets open our bitmap and tell the user
	texPointer = fopen(filename, "r");
	if (texPointer != NULL) {
		printf("readImage: Asset loaded (id): %d\n", texPointer);
	}

	// next, we scan the header to see if the PPM is either a P3 or P6 image. P6 images are useless to us, so exit the process if that happens!
	fscanf(texPointer, "P%c\n", &content);
	if (content != '3') {
		printf("readImage: Must be a P3 ASCII PPM file!\n");
		system("pause");
		exit(-1);
	}

	// in this step, we need to find any comment lines and strip them out
	content = getc(texPointer);
	while (content == '#') {
		do {
			content = getc(texPointer);
		} while (content != '\n');
		content = getc(texPointer);
	}
	ungetc(content, texPointer);

	// this reads our width, height and maxIntensity (used for colour manipulation)
	fscanf(texPointer, "%d", &width);
	fscanf(texPointer, "%d", &height);
	fscanf(texPointer, "%d", &maxIntensity);

	// here, we check if the bitmap is a power of two
	texWidth = allowedSize(width);
	texHeight = allowedSize(height);

	// tell the user the texWidth and texHeight
	printf("readImage: Asset %d is %dx%d\n", texPointer, width, height);

	// now we have these details all sorted, lets push it to our image variable
	image = (pixel *)malloc(texWidth * texHeight * sizeof(pixel));

	// next, we run two for loops. One paints our U-coords to the image var while the other handles V
	for (int itrtr1 = 0; itrtr1 < texHeight; ++itrtr1) {
		for (int itrtr2 = 0; itrtr2 < texWidth; ++itrtr2) {
			fscanf(texPointer, "%d %d %d", &red, &green, &blue);
			k = itrtr1 * texWidth + itrtr2;
			(*(image + k))[0] = (GLubyte)red;
			(*(image + k))[1] = (GLubyte)green;
			(*(image + k))[2] = (GLubyte)blue;
		}
	}

	// lastly, we release our texture from memory
	fclose(texPointer);
}

void loadTexture(const char* filename) {
	GLint level = 0; // the level = no LoD
	GLint components = 3; // 3 components; RGB
	GLint border = 0; // make our texture borderless
	
	//first, read our image file
	printf("loadTexture: Sending details to readImage!\n");
	readImage(filename);

	// change how the engine loads texture variables to suit us
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// define a 2D texture for us to map
	glTexImage2D(GL_TEXTURE_2D, level, components, (GLsizei)texWidth, (GLsizei)texHeight, border, GL_RGB, GL_UNSIGNED_BYTE, image);

	// disable texture wrapping
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// maps pixels, not texels
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
	// set our texture environment to decal, which overwrites pixels with colour
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	
	// lastly, enable texturing
	glEnable(GL_TEXTURE_2D);

	printf("loadTexture: Details retrieved from readImage and texture generated OK!\n");
}

void transform(int direction) {
	// these two if statements determine if the object moves along the X axis or not
	if (direction == 0) { // x+4.0
		rotX += 4.0;
		cout << "rotX: " << rotX << endl;
		display();
	}

	if (direction == 1) { // x-4.0
		rotX -= 4.0;
		cout << "rotX: " << rotX << endl;
		display();
	}

	// these two if statements determine if the object moves along Y or not
	if (direction == 2) { // y+4.0
		rotY += 4.0;
		cout << "rotY: " << rotY << endl;
		display();
	}

	if (direction == 3) { // y-4.0
		rotY -= 4.0;
		cout << "rotX: " << rotY << endl;
		display();
	}

	// these two if statements determine if the object moves along Z or not
	if (direction == 4) { // z+4.0
		rotZ += 4.0;
		cout << "rotZ: " << rotZ << endl;
		display();
	}

	if (direction == 5) { // z-4.0
		rotZ -= 4.0;
		cout << "rotZ: " << rotZ << endl;
		display();
	}
}

void display(void) { // this process loads each frame
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// select our model view matrix and push a new one to the stack
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	// rotate our object on the three axis
	glRotated(rotX, 1.0, 0.0, 0.0);
	glRotated(rotY, 0.0, 1.0, 0.0);
	glRotated(rotZ, 0.0, 0.0, 1.0);
	// load a new object and redisplay
	obj->draw(0.0, 0.0, 0.0);
	// lastly, pop the matrix off our stack
	glPopMatrix();

	// this flips the page
	glutSwapBuffers();
}

void reshape(GLsizei width, GLsizei height) { // this function runs every time we resize our window
	// define our viewport (namely, display widthxheight)
	glViewport(0, 0, width, height);

	// next, load our projection matrix for editing and a fresh identity
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// lastly, set our perspective including field of view
	gluPerspective(50.0 * zoomFactor, ((GLdouble)width) / ((GLdouble)height), 0.01, 10.0);
}

// this processes key presses
void keyboard(unsigned char key, int int1, int int2) {
	if (key == 'w' || key == 'W') { // if w/W is pressed..
		zoomFactor -= 0.1; // .. zoom in

		if (zoomFactor < 0.1) {
			zoomFactor = 0.1;
		}

		reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)); // reload our perspective
		display(); // and draw a new frame
	}

	if (key == 's' || key == 'S') { // if s/S is pressed..
		zoomFactor += 0.1; // .. zoom out

		if (zoomFactor > 1.0) {
			zoomFactor = 1.0;
		}
		
		reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)); // reload our perspective
		display(); // and draw a new frame
	}

	if (key == 'a' || key == 'A') { // if a/A is pressed
		transform(3);
	}

	if (key == 'd' || key == 'D') { // if d/D is pressed
		transform(2); // transform our object (true = clockwise, false = anticlockwise)
	}

	if (key == 'q' || key == 'Q' || key == 0x1b) { // if q/Q or ESC is pressed
		exit(1); // kill
	}
}

// takes in keypresses -- advanced
void special(int key, int int1, int int2) {
	if (key == GLUT_KEY_UP) {
		transform(4);
	}

	if (key == GLUT_KEY_DOWN) {
		transform(5);
	}

	if (key == GLUT_KEY_LEFT) {
		transform(0);
	}

	if (key == GLUT_KEY_RIGHT) {
		transform(1);
	}
}

void timer(int ms) {
	// work out our current framerate and NOW clock value
	tempClock++;
	frameCount++;

	// detect a start time, and a "deltaTime", i.e. the difference between a and b
	//startTime = GetTickCount();
	deltaTime = (startTime - finishTime);

	if (deltaTime > 1000) {
		// print a framerate to console
		printf("framerate: %d\n", frameCount);
		printf("total frames: %d\n", tempClock);

		// get finishTime
		//finishTime = GetTickCount();

		// reset update_count
		frameCount = 0;
	}

	// lastly, recall display and timer funcs
	glutPostRedisplay();
	glutTimerFunc(1000 / FPS, timer, 1000 / FPS);
}

void initialise(void) {
	// material properties matrix
	GLfloat mat_diffuse[] = { 1.0,1.0,0.0,0.0 };

	// our lighting matrix
	GLfloat light_diffuse[] = { 1.0,1.0,1.0,1.0 };

	// and the light's position
	GLfloat position[] = { 1.0,1.0,4.0,1.0 };

	// flat shade our model so we can see the entire thing
	glShadeModel(GL_FLAT);

	// enable normal maps and automate them (we won't be using normal maps but they are required to give our cube definition)
	glEnable(GL_NORMALIZE);
	glEnable(GL_AUTO_NORMAL);

	// set our render background to slate grey
	glClearColor(0.15, 0.3, 0.3, 0.0);


	// load our modelview matrix and a fresh identity
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// the lookat matrix places our camera so we can see our model
	gluLookAt(3.0, 3.0, 3.0, // eyePos
		0.0, 0.0, 0.0, // lookAt
		0.0, 1.0, 0.0); // upVector

	// enable lighting for our scene
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// set our light's position using our previous matricies
	glPushMatrix();
	glLoadIdentity();
	glLightfv(GL_LIGHT0, GL_POSITION, position);
	glPopMatrix();

	// set our material type
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

	// .. and light diffuse
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);

	// enable our depth buffer and clear it for rangefinding
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);

	// load our texture
	loadTexture("assets/tex.ppm");
}

// SDFD209 -- main.cpp -- Jake Deery, 2018
#include "main.h"

int main(int argc, char** argv) {
	int window;

	// say whats going on in console
	printf("Jake's Shit 3D Cube\nSDFD209\nBuild: 070217\n\n=====\n\n");
	printf("Engine 'mainEngine' initialising\n");

	// initialise OpenGL Utilities (glut)
	glutInit(&argc, argv);

	// set our display properties
	glutInitWindowSize(1024, 768);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	// spawn a window
	window = glutCreateWindow("Jake's Shit 3D Cube");
	glutSetWindow(window);

	// run my init function
	initialise();

	// callbacks for glut
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutTimerFunc(1000 / FPS, timer, 1000 / FPS);

	// glut has its own loop procedure! yay!
	glutMainLoop();
}

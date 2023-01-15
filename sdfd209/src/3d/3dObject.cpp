// SDFD209 -- 3dObject.cpp -- Jake Deery, 2018
#include "3dObject.h"

Object3d::Object3d() {

};

Object3d::~Object3d() {

};

void Object3d::draw(float x, float y, float z) {
	glPolygonMode(GL_FRONT, GL_LINE);
	glBegin(GL_TRIANGLES); // begin our triangle list
	// tri1
	glTexCoord2f(0.0625, 0.0); // first, we set our UV map. Like the vertex points, they have to be done clockwise
	glVertex3f(x + 0.0, y + 1.0, z + 0.0); // for the first four triangles, I decided to map the top point, followed by the bottom right, followed by bottom left
	glTexCoord2f(0.125, 1.0);
	glVertex3f(x + 1.0, y + 0.0, z + 1.0);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(x + -1.0, y + 0.0, z + 1.0);
	// tri2
	glTexCoord2f(0.1875, 0.0);
	glVertex3f(x + 0.0, y + 1.0, z + 0.0);
	glTexCoord2f(0.25, 1.0);
	glVertex3f(x + -1.0, y + 0.0, z + 1.0);
	glTexCoord2f(0.125, 1.0);
	glVertex3f(x + -1.0, y + 0.0, z + -1.0);
	// tri3
	glTexCoord2f(0.3125, 0.0);
	glVertex3f(x + 0.0, y + 1.0, z + 0.0);
	glTexCoord2f(0.375, 1.0);
	glVertex3f(x + 1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.25, 1.0);
	glVertex3f(x + 1.0, y + 0.0, z + 1.0);
	// tri4
	glTexCoord2f(0.4375, 0.0);
	glVertex3f(x + 0.0, y + 1.0, z + 0.0);
	glTexCoord2f(0.5, 1.0);
	glVertex3f(x + -1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.375, 1.0);
	glVertex3f(x + 1.0, y + 0.0, z + -1.0);
	// tri5
	glTexCoord2f(0.5, 0.0); // for the last four trianlges, I mapped the top left, followed by the top right, followed by the bottom point
	glVertex3f(x + -1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.625, 0.0);
	glVertex3f(x + -1.0, y + 0.0, z + 1.0);
	glTexCoord2f(0.5625, 1.0);
	glVertex3f(x + 0.0, y + -1.0, z + 0.0);
	// tri6
	glTexCoord2f(0.625, 0.0);
	glVertex3f(x + 1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.75, 0.0);
	glVertex3f(x + -1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.6875, 1.0);
	glVertex3f(x + 0.0, y + -1.0, z + 0.0);
	// tri7
	glTexCoord2f(0.75, 0.0);
	glVertex3f(x + 1.0, y + 0.0, z + 1.0);
	glTexCoord2f(0.875, 0.0);
	glVertex3f(x + 1.0, y + 0.0, z + -1.0);
	glTexCoord2f(0.8125, 1.0);
	glVertex3f(x + 0.0, y + -1.0, z + 0.0);
	// tri8
	glTexCoord2f(0.875, 0.0);
	glVertex3f(x + -1.0, y + 0.0, z + 1.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(x + 1.0, y + 0.0, z + 1.0);
	glTexCoord2f(0.9375, 1.0);
	glVertex3f(x + 0.0, y + -1.0, z + 0.0);
	glEnd();
}

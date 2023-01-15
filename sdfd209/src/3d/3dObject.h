// SDFD209 -- 3dObject.h -- (C) Jake Deery, 2017
#pragma once
#ifndef object3d_h
#define object3d_h

// include my global header
#include "global.h"

class Object3d {
public:
	Object3d();
	~Object3d();

	void draw(float x, float y, float z);
private:

};

#endif

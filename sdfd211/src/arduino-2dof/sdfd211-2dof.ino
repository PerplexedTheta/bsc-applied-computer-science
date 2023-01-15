#include <Servo.h>
void invK(float tempX, float tempY);

float x = 40;
float y = 120;
float l1 = 105;
float l2 = 121;
float e = 0;
float s = 0;
float ang = 0;

int a = 40;//x start
int b = 120;//x end
int c = 120;
int d = 180;

Servo servo1;
Servo servo2;

void setup()	{
	Serial.begin(9600);
	servo1.attach(13);
	servo2.attach(12);
}

void loop() {
	// top left -> right
	for(x = a; x < b; x++) invK(x,y); // 80w
	
	// top right V bottom right
	for(y = c; y < d; y++) invK(x,y); // 40h
	
	// bottom right > bottom left
	for(x = b; x > a; x--) invK(x,y); // 80w
	
	// bottom left ^ top left
	for(y = d; y > c; y--) invK(x,y); // 40h
	
	// top left > mid-top
	for(x = a; x < 80; x++) invK(x,y); // 40w
	
	// circle
	for (int i = 0; i < 200; i++) {
		ang = (float)i * 2.0 * 3.141592 / 200.0;
		
		x = 80.0 - sin(ang) * 40.0;
		y = 150.0 - cos(ang) * 30.0;
		
		invK(x,y);
	}

	// s -- top semi
	for (int i = 0; i < 200; i++) {
		ang = (float)i * 2.0 * 3.141592 / 400.0;
		
		x = 80.0 - sin(ang) * 15.0;
		y = 135.0 - cos(ang) * 15.0;
		
		invK(x,y);
	}

	// s -- bottom semi
	for (int i = 0; i < 200; i++) {
		ang = ((2.0 * 3.141592)/400.0) * (float)i;
		
		x = 80 + sin(ang) * 15;
		y = 165 - cos(ang) * 15;
		
		invK(x,y);
	}
}

void invK(float tempX, float tempY) {
	e = degrees(acos((tempX*tempX+tempY*tempY-l1*l1-l2*l2) / (2.0*l1*l2)));
	s = degrees(atan(tempY/tempX) - acos((tempX*tempX+tempY*tempY+l1*l1-l2*l2) / (2*l1*sqrt(tempX*tempX+tempY*tempY))));
	
	Serial.println(e, 5);
	Serial.println(s, 5);
	
	servo1.write(e);
	delay(10);
	servo2.write(s);
	delay(10);
}

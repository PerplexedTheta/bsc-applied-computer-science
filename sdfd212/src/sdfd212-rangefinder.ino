// pins: input
int red = 8;
int amber = 9;
int green = 10;
int buzzer = 11;
int trig = 12;

// pins: output
int button = 2;
int echo = 13;

// global vars
bool ignition = true;
int rangeFindDist = 0;

void setup() {
	// set baud rate
	Serial.begin(57600);
	
	// set pins
	pinMode(LED_BUILTIN, OUTPUT);
	pinMode(red, OUTPUT);
	pinMode(amber, OUTPUT);
	pinMode(green, OUTPUT);
	pinMode(trig, OUTPUT);
	pinMode(buzzer, OUTPUT);
	pinMode(button, INPUT);
	pinMode(echo, INPUT);
	
	// all lights on and wait (test)
	Serial.print("TEST");
	Serial.println("");
	digitalWrite(LED_BUILTIN, 1);
	digitalWrite(red, 1);
	digitalWrite(amber, 1);
	digitalWrite(green, 1);
	digitalWrite(buzzer, 1);
	delay(500);
	
	// all off
	Serial.print("all off");
	Serial.println("");
	digitalWrite(LED_BUILTIN, 0);
	digitalWrite(red, 0);
	digitalWrite(amber, 0);
	digitalWrite(green, 0);
	digitalWrite(buzzer, 0);
}

void loop() {
	// detect button press
	if (digitalRead(button) == 1) {
		// disable EVERYTHING
		digitalWrite(LED_BUILTIN, 0);
		digitalWrite(red, 0);
		digitalWrite(amber, 0);
		digitalWrite(green, 0);
		digitalWrite(buzzer, 0);
		
		// set ignition
		ignition = false;
		Serial.print("ignition == ");
		Serial.print(ignition);
		Serial.println("");
		delay(250);
	}
	
	while (ignition == false) {
		delay(250);
		
		// detect button press
		if (digitalRead(button) == 1) {
			// enable onboard LED
			digitalWrite(LED_BUILTIN, 1);
			
			// reset ignition
			ignition = true;
			Serial.print("ignition == ");
			Serial.print(ignition);
			Serial.println("");
			delay(250);
			break;
		}
	}
	// get the range from the ultrasound
	rangeFindDist = rangeFind();

	// set the LED
	if(rangeFindDist > 99) {
		digitalWrite(red, 0);
		digitalWrite(amber, 0);
		digitalWrite(green, 1);

		// make sure the buzzer is off
		digitalWrite(buzzer, 0);
	} else if(rangeFindDist > 49 && rangeFindDist < 100) {
		digitalWrite(red, 0);
		digitalWrite(amber, 1);
		digitalWrite(green, 0);

		// make sure the buzzer is off
		digitalWrite(buzzer, 0);
	} else if(rangeFindDist < 50) {
		digitalWrite(red, 1);
		digitalWrite(amber, 0);
		digitalWrite(green, 0);

		// annoy andy
		digitalWrite(buzzer, 1);
	}
}

int rangeFind() {
	// ultrasound vars
	long duration = 0;
	int distance = 0;

	// Clears the trig pin
	digitalWrite(trig, 0);
	delayMicroseconds(2);
	
	// 10μs burst
	digitalWrite(trig, 1);
	delayMicroseconds(10);
	digitalWrite(trig, 0);
	
	// reads the echo, returns the sound wave travel time in microseconds
	duration = pulseIn(echo, 1);
	
	// compute distance
	distance = duration * 0.034 / 2;
	
	// print debug
	Serial.print("distance == ");
	Serial.print(distance);
	Serial.println("");

	return distance;
}

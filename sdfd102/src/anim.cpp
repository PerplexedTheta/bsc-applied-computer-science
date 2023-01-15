// anim.cpp -- Jake Deery 2016
#include "anim.h"

int anim() { // Function for the interactive animation
	clock1 = GetTickCount(); // Checks the PC time
	do { // Next, set up a while loop that only ends when key_pressed is ESC
		clock2 = GetTickCount(); // Checks the PC time
		if (clock2 - clock1 > 50) { // If the difference between clock2 and clock1 is greater than 100...
			clock1 = clock2; // ...set clock1 to the same value as clock2
			if (KEY_DOWN(VK_UP)) { // If the key down is vk_up...
				y--; // ... incriment down y
			}

			if (KEY_DOWN(VK_DOWN)) { // If the key down is vk_down...
				y++; // ... incriment up y
			}

			if (KEY_DOWN(VK_LEFT)) { // If the key down is vk_left...
				x--; // ... incriment down x
			}

			if (KEY_DOWN(VK_RIGHT)) { // If the key down is vk_right...
				x++; // ... incriment up x
			}

			if (x > (SW-10)) { // if x is greater than (SW-10)...
				x = (SW-10); // ... set x to (SW-10)
			}
			
			if (y > (SH-14)) { // if y is greater than (SH-14)...
				y = (SH-14); // ... set y to (SH-14)
			}

			if (x < 1) { // if x is less than 1...
				x = 1; // ... set x to 1
			}

			if (y < 1) { // if y is less than 1...
				y = 1; // ... set y to 1
			}

			clrscr(); // Clear the last draw...
			drawMark(x, y); // Draw mark again with the new x and y values
		}
	} while (!KEY_DOWN(VK_ESCAPE)); // When ESC is pressed...

	mainMenu();
	return 0;
}

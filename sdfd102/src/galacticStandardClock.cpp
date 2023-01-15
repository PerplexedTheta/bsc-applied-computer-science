// galacticStandardClock.cpp -- Jake Deery 2016
#include "galacticStandardClock.h"

bool isRunning = true; // Basically this bool just states whether or not the application will run

int galacticStandardClock() { // A clock that counts up in mysterious ways (488 ms = 1 sec, 100 sec = 1 min, 100 min = 1 hr, 20 hrs = 1 day)
	clrscr();
	isRunning = true;

	gotoxy(0,0);
	std::cout << "Galactic Standard Clock"; // Explain what the clock is
	gotoxy(0,1);
	std::cout << "Press (R) to reset, or (ESC) to exit the program." << std::endl << std::endl;

	gotoxy(0,5);
	std::cout << "* Clock rules:" << std::endl; // Start off by saying the 'clock rules'
	gotoxy(0,6);
	std::cout << "* One second is equal to 500 Earth miliseconds" << std::endl; // Explain what one second is
	gotoxy(0,7);
	std::cout << "* One minute equals 100 seconds (50 Earth secs)" << std::endl; // Explain what one minute is
	gotoxy(0,8);
	std::cout << "* One hour equals 100 minutes (83 Earth mins, 20,4 secs)" << std::endl; // Explain what one hour is is
	gotoxy(0,9);
	std::cout << "* One day equals 20 hours (27 Earth hrs, 46 mins, 4,8 secs)" << std::endl; // Explain what one day is is

	for (int hours = 0; hours<20 && isRunning; hours++) { // Third loop in nest: If hours are less than 20, incriment hours. Else make hours 0 and restart process.
		for (int minutes = 0; minutes<100 && isRunning; minutes++) { // Second loop in nest: If minutes are less than 100, incriment minutes. Else make minutes 0 and restart process.
			for (int seconds = 0; seconds<100 && isRunning; seconds++) { // First loop in nest: If seconds are less than 100 in total, incriment seconds. Else make seconds 0 and restart process.
				gotoxy(0,3);
				if (seconds<10 && minutes<10 && hours<10) { // If seconds is less than 10, minutes is less than 10 and hours is less than 10...
					std::cout << "0" << hours << ":0" << minutes << ":0" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				} else if (minutes<10 && hours<10) { // If minutes is less than 10 and hours is less than 10...
					std::cout << "0" << hours << ":0" << minutes << ":" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				} else if (hours<10) { // If hours is less than 10...
					std::cout << "0" << hours << ":" << minutes << ":" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				} else if (minutes<10 && seconds<10) { // If minutes is less than 10 and seconds is less than 10...
					std::cout << hours << ":0" << minutes << ":0" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				} else if (seconds<10) {// If seconds is less than 10...
					std::cout << hours << ":" << minutes << ":0" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				} else { // If none of the rules above are true...
					std::cout << hours << ":" << minutes << ":" << seconds << std::endl << std::endl; // cout the result of all the loops thus far
				}

				if (KEY_DOWN(VK_ESCAPE)) {// If esc has been pressed...
					isRunning = false; // Set the state of isRunning to false so all the loops close on the next cycle
				}

				if (KEY_DOWN(0x52)) { // If r has been pressed...
					hours = 0; // ... reset hours, mins and secs
					minutes = 0;
					seconds = 0;
				}

				Sleep(500); // Pause the loop for 500 ms
			}
		}
	}

	extrasMenu();
	return 0;
}

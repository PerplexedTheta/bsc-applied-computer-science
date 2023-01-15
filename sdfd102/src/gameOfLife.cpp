// gameOfLife.cpp -- Jake Deery 2016
#include "gameOfLife.h"

int primary[SW][SH]; // Create array named int a
int secondary[SW][SH]; // ... and int z
int detectedLiveBlocks = 0; // and one called c, specifically set at 0

int gameOfLife() { // A Game of Life
	textcolor(15);
	//randomly set array a
	for (x = 0; x < (SW - 1); x++) { // Starting at 0, if x is less than (SW-1) (def. 119), incriment x by one.
		for (y = 0; y < (SH - 1); y++) { // Starting at 0, if y is less than (SH-1) (def. 29), incriment y by one.
			primary[x][y] = rand() % 2; // Randomise the value of primary[]
		}
	}

	clock1 = GetTickCount(); // Checks the PC time
	do { // Next, set up a while loop that only ends when key_pressed is ESC
		clock2 = GetTickCount(); // Checks the PC time
		if (clock2 - clock1 > 500) { // If the difference between clock2 and clock1 is greater than 100...
			clock1 = clock2; // ...set clock1 to the same value as clock2
			clrscr();

			//view array a
			for (x = 0; x < (SW - 1); x++) { // Starting at 0, if x is less than (SW-1) (def. 119), incriment x by one.
				for (y = 0; y < (SH - 1); y++) { // Starting at 0, if y is less than (SH-1) (def. 29), incriment y by one.
					gotoxy(x, y); // Move the cursor to the values of X and Y,
					if (primary[x][y] == 1) { // and if the value of x,y is 1 in the a array...
						std::cout << block; // ... print b
					}
				}
			}

			//calculate next generation
			for (x = 1; x < (SW - 1); x++) { // Starting at 1, if x is less than (SW-1), incriment x
				for (y = 1; y < (SH - 1); y++) { // Starting at 1, if y is less than (SH-1), incriment y
					detectedLiveBlocks = 0; // Set c to 0...
					if (primary[x - 1][y - 1] == 1) { // if top left = 1... 
						detectedLiveBlocks++; // incriment c...
					}
					if (primary[x][y - 1] == 1) {//top middle
						detectedLiveBlocks++;
					}
					if (primary[x + 1][y - 1] == 1) { //top right
						detectedLiveBlocks++;
					}
					if (primary[x - 1][y] == 1) { //middle left
						detectedLiveBlocks++;
					}
					if (primary[x + 1][y] == 1) { //middle right
						detectedLiveBlocks++;
					}
					if (primary[x - 1][y + 1] == 1) { //bottom left
						detectedLiveBlocks++;
					}
					if (primary[x][y + 1] == 1) { //bottom middle
						detectedLiveBlocks++;
					}
					if (primary[x + 1][y + 1] == 1) { //bottom right
						detectedLiveBlocks++;
					}

					// game of life rules
					if (primary[x][y] == 1 && detectedLiveBlocks < 2) { //The first rule . If c is 2 or less and a cell is alive, kill it
						secondary[x][y] = 0;
					}
					if (primary[x][y] == 1 && (detectedLiveBlocks == 2 || detectedLiveBlocks == 3)) { //The second Rule . If c is 2 or 3, and a cell is alive, leave it alive
						secondary[x][y] = 1;
					}
					if (primary[x][y] == 1 && detectedLiveBlocks > 3) { //The thrid	rule . If c is more than 3 and a cell is alive, kill it
						secondary[x][y] = 0;
					}
					if (primary[x][y] == 0 && detectedLiveBlocks == 3) { //The final rule. If c is 3 and a cell is dead, resurrect it
						secondary[x][y] = 1;
					}
				}
			}

			// copy a to z
			for (x = 0; x < (SW - 1); x++) { // Starting at 0, if x is less than (SW-1) (def. 119), incriment x by one.
				for (y = 0; y < (SH - 1); y++) { // Starting at 0, if y is less than (SH-1) (def. 29), incriment y by one.
					primary[x][y] = secondary[x][y];
				}
			}

			if (KEY_DOWN(0x47)) { // If G is pressed...
				clrscr(); // ... clear the screen
				for (x = 0; x < (SW - 1); x++) { // Starting at 0, if x is less than (SW-1) (def. 119), incriment x by one.
					for (y = 0; y < (SH - 1); y++) { // Starting at 0, if y is less than (SH-1) (def. 29), incriment y by one.
						primary[x][y] = 0; // Make primary[] 0...
						secondary[x][y] = 0; // /// and the same for secondary[]
					}
				}

				primary[2][6] = 1; // The following section will now draw a gosper glider
				primary[2][7] = 1;
				primary[3][6] = 1;
				primary[3][7] = 1;
				primary[12][6] = 1;
				primary[12][7] = 1;
				primary[12][8] = 1;
				primary[13][5] = 1;
				primary[13][9] = 1;
				primary[14][4] = 1;
				primary[15][4] = 1;
				primary[14][10] = 1;
				primary[15][10] = 1;
				primary[16][7] = 1;
				primary[17][5] = 1;
				primary[17][9] = 1;
				primary[18][6] = 1;
				primary[18][7] = 1;
				primary[18][8] = 1;
				primary[19][7] = 1;
				primary[22][4] = 1;
				primary[23][4] = 1;
				primary[22][5] = 1;
				primary[23][5] = 1;
				primary[22][6] = 1;
				primary[23][6] = 1;
				primary[24][3] = 1;
				primary[24][7] = 1;
				primary[26][2] = 1;
				primary[26][3] = 1;
				primary[26][7] = 1;
				primary[26][8] = 1;
				primary[35][4] = 1;
				primary[36][4] = 1;
				primary[35][5] = 1;
				primary[36][5] = 1;
			}

			if (KEY_DOWN(0x20)) { // If spacebar is pressed...
				//randomly set array a
				for (x = 0; x < (SW - 1); x++) { // Starting at 0, if x is less than (SW-1) (def. 119), incriment x by one.
					for (y = 0; y < (SH - 1); y++) { // Starting at 0, if y is less than (SH-1) (def. 29), incriment y by one.
						primary[x][y] = rand() % 2; // Randomise the value of primary[]
					}
				}
			}
		}
	} while (!KEY_DOWN(VK_ESCAPE)); // If VK_ESCAPE is pressed, exit the do/while

	mainMenu();
	return 0;
}

// andyCode.cpp -- Andy Watson @ CityPlym -- Modified!
#include "andyCode.h"

int hline(int row, int column, int width) { // hline function
	for (int iterator = 0; iterator < width; iterator++) { // Starting at 0, make an iterator that counts until the iterator is greater than the width
		gotoxy(iterator + row, column); // Gotoxy, where the x co-ord is the row the user asked for + the iterator and the column remains the same...
		std::cout << block; // ... and print out a block
	}

	return 0;
}

int vline(int row, int column, int height) { // vline function
	for (int iterator = 0; iterator < height; iterator++) { // Starting at 0, make an iterator that counts until the iterator is greater than the height
		gotoxy(row, iterator + column); // Gotoxy, where the x co-ord is the row the user asked for + the iterator and the column remains the same...
		std::cout << block; // ... and print out a block
	}

	return 0;
}

int textcolor(int colours) { // textcolor function. This basically just changed the colour of the text after specified.
	HANDLE hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, colours);

	return 0;
}

int gotoxy(int row, int column) { // gotoxy function. This basically moved the cursor to the specified location.
	static HANDLE hStdout = NULL;
	COORD coord;

	coord.X = row;
	coord.Y = column;
	if (!hStdout) {
		hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	}

	SetConsoleCursorPosition(hStdout, coord);

	return 0;
}

int clrscr() { // clrscr function. This basically just clears the screen and moves the cursor back to [0,0]
	static HANDLE hStdout = NULL;
	static CONSOLE_SCREEN_BUFFER_INFO csbi;
	const COORD startCoords = { 0,0 };
	DWORD dummy;

	if (!hStdout)
	{
		hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hStdout, &csbi);
	}
	FillConsoleOutputCharacter(hStdout, ' ', csbi.dwSize.X * csbi.dwSize.Y, startCoords, &dummy);
	gotoxy(0, 0);

	return 0;
}

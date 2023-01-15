// drawMark.cpp -- Jake Deery 2016
#include "drawMark.h"

int drawMark(int x, int y) { // Draws a jar of marmalade
	textcolor(15);// Colour white
	hline(x + 1, y + 1, 8); // The lid
	textcolor(4); // Colour red
	hline(x + 2, y + 2, 6); // 1st line MARMALADE
	hline(x + 2, y + 3, 6); // 2nd line MARMALADE
	hline(x + 2, y + 4, 6);// 3rd line MARMALADE
	textcolor(15); // Colour white
	hline(x + 2, y + 5, 6); // 4th line GOURMET PAPERE
	hline(x + 2, y + 6, 6); // 5th line GOURMET PAPERE
	hline(x + 2, y + 7, 6); // 6th line GOURMET PAPERE
	hline(x + 2, y + 8, 6); // 7th line GOURMET PAPERE
	textcolor(4); // Colour red
	hline(x + 2, y + 9, 6); // 8th line MARMALADE
	hline(x + 2, y + 10, 6); // 9th line MARMALADE
	hline(x + 2, y + 11, 6); // 10th line MARMALADE
	hline(x + 3, y + 12, 4); // Jar bottom
	textcolor(0); // Colour black
	gotoxy(x + 3, y + 5); // Left eye
	std::cout << block;
	gotoxy(x + 6, y + 5); // Right eye
	std::cout << block;
	gotoxy(x + 2, y + 7); // Left lip corner
	std::cout << block;
	gotoxy(x + 7, y + 7); // Right lip corner
	std::cout << block;
	hline(x + 3, y + 8, 4); // Mouth
	textcolor(13);
	gotoxy(x + 2, y + 6); // Left cheek
	std::cout << block;
	textcolor(13);
	gotoxy(x + 7, y + 6); // Right cheek
	std::cout << block;
	gotoxy(0, 0); // Reset the position of the cursor

	return 0;
}

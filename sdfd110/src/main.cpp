// main.cpp -- Jake Deery 2017
#include "main.h"

int main() {
	printf("Assembly 04 -- 2017/02/11 \02\n");
	printf("\nAdd two integers together \02\n");
	__asm {
		mov edx, 0x2
		add edx, 0x7 // Data register should total 9
	}
	system("pause");
	printf("\nSubtract two integers together \02\n");
	__asm {
		mov edx, 0x7
		sub edx, 0x2 // Data register should total 5
	}
	system("pause");
	printf("\nA for/next loop in assembly \02\n");
	__asm {
		mov eax, 0x0 // Set the accumulator to 0
		mov ecx, 0x7 // Set the counter to 7
		loop_start: // Start the loop
		add eax, ecx // Add the value of the counter to the value of the accumulator
		loop loop_start // Go back to loop_start (counter--)
		// Final output should be: 0x1C
		// 
		// Pseudo-code for this assembly code:
		// sum = 0;
		// for (i=10; i>0; i−−) {
		// sum += i;
		// }
	}
	system("pause");
	printf("\nAn if/else statement in assembly \02\n");
	__asm {
		cmp eax, 0x0 // IF accumulator == 0, set ZF
		jz thenblock // Goto thenblock IF ZF is set
		mov edx, 0x2 // Move 2 to base register
		jmp next // Goto next because we don't want to run the THEN
		thenblock :
		mov edx, 0x1 // THEN move 1 to the base register
		next:
		// Final output should be: If eax > 0, set edx to 2 and pass. Else set edx to 1 and pass.
		// 
		// Pseudo-code for this assembly code:
		// if (EAX == 0) {
		// EBX = 1;
		// } else {
		// EBX = 2;
		// }
	}
	system("pause");
	printf("\nMultiply two integers together \02\n");
	__asm {
		mov eax, 0x2
		mov edx, 0x7
		mul edx // Accumulator should equal 0xE
	}
	system("pause");
	printf("\nMultiply two signed integers together \02\n");
	__asm {
		mov eax, 0x2
		mov edx, 0x7
		imul edx // Accumulator should equal 0xE
	}
	system("pause");
	printf("\nDivide two integers together \02\n");
	__asm {
		mov edx, 0x0 // Empty the data register
		mov eax, 0x7
		mov ebx, 0x2
		div ebx // Accumulator should equal 0x3, base register should equal 0x1
	}
	system("pause");
	printf("\nThis process has halted sucessfully! \02");
	return 0;
}

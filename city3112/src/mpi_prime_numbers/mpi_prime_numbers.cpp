// mpi_prime_numbers.cpp, jake deery, 2020
#include "mpi_prime_numbers.h"

// main process
int main(int argc, char *argv[]) {
	// vars
	int numberOfProcesses;
	int processId;
	int minNumber = 1;
	int maxNumber = 1048576;
	int number = minNumber;
	int numberFactor = 2;
	int primesChunk;
	int primes;
	double deltaTime;

	// init mpi
	if(MPI_Init(&argc, &argv) != 0) {
		cout << "\n";
		cout << "[E] FATAL ERROR" << "\n";
		cout << "[E] MPI_Init returns non-zero code. Please check your setup and code and try again!" << "\n";
		return(1);
	}

	// fetch number of processes
	if(MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses) != 0) {
		cout << "\n";
		cout << "[E] FATAL ERROR" << "\n";
		cout << "[E] MPI_Comm_size returns non-zero code. Please check your setup and code and try again!" << "\n";
		return(1);
	}

	// fetch process rank for this process
	if(MPI_Comm_rank(MPI_COMM_WORLD, &processId) != 0) {
		cout << "\n";
		cout << "[E] FATAL ERROR" << "\n";
		cout << "[E] MPI_Comm_rank returns non-zero code. Please check your setup and code and try again!" << "\n";
		return(1);
	}

	// show init
	if(processId == 0) {
		cout << "\n";
		cout << "=============================================" << "\n";
		cout << "= PRIME NUMBER CALCULATOR, JAKE DEERY, 2020 =" << "\n";
		cout << "=============================================" << "\n";
		cout << "	" << numberOfProcesses << " processes are available" << "\n";
		cout << " N		Pi		Time" << "\n";
	}

	// main loop
	while (number < maxNumber + 1) {
		if(processId == 0) deltaTime = MPI_Wtime(); // start clock

		// broadcast number
		if(MPI_Bcast (&number, 1, MPI_INT, 0, MPI_COMM_WORLD) !=0 ) {
			cout << "\n";
			cout << "[E] FATAL ERROR" << "\n";
			cout << "[E] MPI_Bcast returns non-zero code. Please check your setup and code and try again!" << "\n";
			return(1);
		}

		// define problem for mpi
		primesChunk = testPrimeNumber(number, processId, numberOfProcesses);

		// reduce down problems
		if((MPI_Reduce(&primesChunk, &primes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD )) !=0 ) {
			cout << "\n";
			cout << "[E] FATAL ERROR" << "\n";
			cout << "[E] MPI_Reduce returns non-zero code. Please check your setup and code and try again!" << "\n";
			return(1);
		}

		// finalise on primary node
		if (processId == 0) {
			deltaTime = MPI_Wtime() - deltaTime;
      cout << " " << number << "		" << primes << "		" << deltaTime << "\n";
    }

		// basically, double the number
    number = number * numberFactor;
  }

	// kill
	if(MPI_Finalize() != 0) {
		cout << "\n";
		cout << "[E] FATAL ERROR" << "\n";
		cout << "[E] MPI_Finalize returns non-zero code. Please check your setup and code and try again!" << "\n";

		return 1;
	} else if (processId == 0) {
		cout << "\n";
		cout << "[I] Process complete" << "\n";

		return 0;
	}
}


// do the calculations
int testPrimeNumber(int number, int processId, int numberOfProcesses) {
	// vars
	int i;
	int j;
  int prime;
  int total = 0;

	// start solving primes
  for (i = 2 + processId; i < number + 1; i = i + numberOfProcesses) {
    prime = 1;
    for (j = 2; j < i; j++) {
      if(( i % j ) == 0) {
				prime = 0;
        break;
      }
    }
    total = total + prime;
  }

	// deliver result
  return total;
}

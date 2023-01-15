// mpi_prime_numbers.h, jake deery, 2020

#pragma once

#ifndef MPIPRIMES
#define MPIPRIMES

// defines
#define ARRAYSIZE 1000

// specift h files
#include <iostream>
#include <mpi.h>

// define null as always 0
#ifndef NULL
#define NULL 0
#endif

// userspace
using std::cout;

// define funcs
extern int testPrimeNumber(int number, int processId, int numberOfProcesses);

#endif

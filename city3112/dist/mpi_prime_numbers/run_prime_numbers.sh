#!/bin/sh
mpirun.openmpi -np 12 --hostfile ../hostfile ~/prime_numbers/mpi_prime_numbers

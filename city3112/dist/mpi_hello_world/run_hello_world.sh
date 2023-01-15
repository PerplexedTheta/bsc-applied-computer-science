#!/bin/sh
mpirun.openmpi -np 12 --hostfile ../hostfile ~/hello_world/mpi_hello_world

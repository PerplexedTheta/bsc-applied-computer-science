#!/bin/sh

# vars
np=4 # num. processors

# main loop
for i in 1 2 3 # do three linpack tests, to get good average
do
	echo "Running cycle $i, please be patient . . . "
	mpirun.openmpi -np $np --hostfile ../hostfile /bin/hpcc & # main proc
	wait # wait for execution before continuing
	mv hpccoutf.txt hpccoutf-$i.txt # backup results
done

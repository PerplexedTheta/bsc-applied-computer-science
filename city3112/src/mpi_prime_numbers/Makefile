EXECS=mpi_prime_numbers
MPICC?=mpic++

all: ${EXECS}

mpi_prime_numbers: mpi_prime_numbers.cpp
	${MPICC} -lgmp -o mpi_prime_numbers mpi_prime_numbers.cpp

clean:
	rm ${EXECS}

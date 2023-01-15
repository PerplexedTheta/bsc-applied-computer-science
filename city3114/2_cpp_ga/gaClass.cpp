// gaClass.cpp, city3114_proc2, jake deery, 2020
#include "gaClass.h"

gaClass::gaClass() {
	// init
	cout << "[I] Initialising ga object . . . done" << "\n";
	cout << "\n";
}

gaClass::~gaClass() {
	// bye
	cout << "[I] Deleting ga object . . . done" << "\n";
	cout << "[I] Program quitting . . . done" << "\n";
}

int gaClass::runProc() {
	// init
	initPopulation();//create a random population of individuals
	
	// do proc -- eggholder version
	for(currentGen = 0; currentGen < generations; currentGen++) {
		getFitness(0, 1024.0); // args: fitnessFunction, range
		viewPopulation();
		doCrossover();
		mutate();
		swapPopulation();
	}
	
	return 0;
}

int gaClass::initPopulation() {
	for(int32_t i = 0; i < populationSize; i++) {
		for (int32_t j = 0; j < individual; j++) {
			ind[i][j] = rand() % 2;
		}
	}
	
	return 0;
}

int gaClass::getFitness(int32_t fitnessFunction, float64_t range) {
	// vars
	int32_t sum1 = 0;
	int32_t sum2 = 0;
	int32_t z = 0;
	int32_t ms = 0;
  float64_t min = 0.0;
	float64_t x = 0.0;
	float64_t y = 0.0;
	float64_t tt = 0.0;

  for (int32_t i = 0; i < populationSize; i++) {
		// this lump of code gets the x bits' bits (ooh err)
		sum1 = 0;
		z = 1;
		
		for (int32_t j = 0; j < (individual / 2); j++) {
			if (ind[i][(individual / 2) - j - 1] == 1) sum1 += z;
			z *= 2;
		}
		
		// this lump of code gets the y bits' bits
		sum2 = 0;
		z = 1;
		
		for (int32_t j = 0; j < (individual / 2); j++) {
			if (ind[i][individual - j - 1] == 1) sum2 += z;
			z *= 2;
		}
		
		// now normalise each value (range 0.000-1.000)
		x = sum1 / pow(2, (individual / 2));
		y = sum2 / pow(2, (individual / 2));
		
		// and finally multiply by the range and sub range/2 to get 'bounding box'
		x = x * range - (range / 2);
		y = y * range - (range / 2);
		
		// finally calculate fitness from eggholder function
		// 0 == eggholder
		// 1 == schaffer n.2
		if(fitnessFunction == 0) fitness[i] = -1.0 * (y + 47.0) * sin(sqrt(abs(y + x / 2.0 + 47.0))) + -1.0 * x * sin(sqrt(abs(x - (y + 47.0))));
		if(fitnessFunction == 1) fitness[i] = 0;
		
		zz++;
	}
	
	//now sort in acending order
	//this is a simple selection sort
	for (int32_t j = 0; j < populationSize; j++) {
		min = fitness[j];//set a minimum to be the first element in the list
		ms = j;//set minimum value array index
		
		for (int32_t i = j + 1; i < populationSize; i++) { // loop to check for minimum
			if (fitness[i] < min) {
				min = fitness[i];
				ms = i;//save the array position of the minimum slot
			}
		}
		
		//end of i loop
		//swap fitness
		fitness[ms] = fitness[j];
		fitness[j] = min;
		
		//swap individual
		for (int32_t i = 0; i < individual; i++) temp[i] = ind[ms][i];
		for (int32_t i = 0; i < individual; i++) ind[ms][i] = ind[j][i]; 
		for (int32_t i = 0; i < individual; i++) ind[j][i]=temp[i];
	}
	
	//end of j loop
	//now normalise the fitness
	
	tt = fitness[0];//save the lowest fitness
	for (int32_t i = 0; i < populationSize; i++) {
		normFit[i] = fitness[i]-tt;
	}
	
	tt = normFit[populationSize - 1];
	for (int32_t i = 0; i < populationSize; i++) {
		normFit[i] = tt-normFit[i];
	}
	
	totalFit = 0.0;
	for (int32_t i = 0; i < populationSize; i++) {
		totalFit = totalFit+ normFit[i];
	}
	
	return 0;
}

int gaClass::viewPopulation() {
	cout << "[I] Current generation: " << currentGen+1 << "\t\tComputational effort: " << zz << "\n";
	
	for (int32_t i = 0; i < 10; i++) {
		cout << "Individual " << i << ": ";
		
		for (int32_t j = 0; j < individual; j++) {
			cout << ind[i][j];
		}
		
		cout << "\t\tCurrent fitness=" << fitness[i] << "    \tnormFit value: " << normFit[i] << "\n";
	}
	
	return 0;
}

int gaClass::doCrossover() {
	// vars
	int32_t numcross = 0;
	int32_t ss = 0;
	int32_t s1 = 0;
	int32_t s2 = 0;
	int32_t crosspoint = 0;
	float64_t rnum = 0.0;
	float64_t tempfit = 0.0;
	
	//copy over the elite without mutation or crossover
	for (int32_t i = 0; i < 1; i++) {
		for (int32_t j = 0; j < individual; j++) newInd[i][j] = ind[i][j];
	}

	//now perform crossover 
	//we use the normfit[] array for roulette wheel selection
	//first caclulate how many children will be created
	numcross = (int32_t)(crossover * populationSize);
  
	for (int32_t i = 1; i < numcross; i+=2) {
		//select parent 1 stored in s1
		rnum = ((float64_t)rand()/RAND_MAX) * totalFit;
		s1 = -1;
		tempfit = 0.0;
		
		while (tempfit < rnum) {
			s1++;
			tempfit = tempfit+ normFit[s1];
		}
		
		//select parent 2 stored in s2
		rnum = ((double)rand()/RAND_MAX) * totalFit;
		s2 = -1;
		tempfit = 0.0;
		
		while (tempfit < rnum) {
			s2++;
			tempfit = tempfit+ normFit[s2];
		}

		//select crossover point
		crosspoint = 1 + rand() % (individual - 1);
		
    //now create the two children
    for(int32_t j = 0; j < crosspoint; j++) {
			newInd[i][j] = ind[s1][j];
			newInd[i+1][j] = ind[s2][j];
		}
		
		for(int32_t j = crosspoint; j < individual; j++) {
			newInd[i][j] = ind[s2][j];
			newInd[i+1][j] = ind[s1][j];
		}
		//generate child1 and child 2
	}
	
	//finally fill the rest of the population with roulette wheel selection
	for (uint32_t i = numcross; i < populationSize; i++) {
		//select individual
		//first create random number up to totfit
		rnum = ((double)rand()/RAND_MAX) * totalFit;
		ss = -1;
		tempfit = 0.0;
		
		 while (tempfit < rnum) {
			ss++;
			tempfit = tempfit+ normFit[ss];
		 }
		
		//copy into newpop
		for (int32_t j = 0; j < individual; j++) newInd[i][j] = ind[ss][j];
	}
	//at this point the next generation is full and stored in newind[][]
	
	return 0;
}

int gaClass::mutate() {
	// vars
	int32_t z = 0;
	
	for (int32_t i = 1; i < populationSize; i++) {
		if ((double)(rand() / RAND_MAX) < mutation) {
			z = rand() % individual;
			newInd[i][z] = !newInd[i][z];
		}
	}
	
	return 0;
}

int gaClass::swapPopulation() {
	for (int i = 0; i < populationSize; i++) {
		for (int j = 0; j < individual; j++) {
			ind[i][j] = newInd[i][j];
		}
	}
	
	return 0;
}

// gaClass.h, city3114_proc2, jake deery, 2020
#pragma once
#ifndef GACLASS_H
#define GACLASS_H

// include
#include <iostream>
#include <math.h>

// typedef - float
typedef float float32_t;
typedef double float64_t;
typedef long double float96_t;

// namespace
using std::cout;
using std::cin;



// class def
class gaClass {
	public:
		// vars
		
		// methods
		gaClass();
		~gaClass();
		
		int runProc();

	private:
		// consts -- change these to change what happens
		// this is filthy and if i gave myself more time this would be less crappy
		static const int32_t individual = 20 * 2;			// total bits for one individual (bits * dims)
		static const int32_t populationSize = 1000;		// no. of individuals
		static const int32_t generations = 100;				// max. gens
		const float32_t mutation = 0.001;							// mutation probability -- eg environment hostility
		const float32_t crossover = 0.6;							// crossover probability -- i.e. how many parents die to be replaced by their children
																									// note: the population size always remains the same, so if a parent does not die, the
																									// child must be discarded (eaten?)
		
		// vars -- utility
		int32_t currentGen = 0;
		int32_t zz = 0;
		float64_t totalFit = 0.0;
		
		int32_t ind[populationSize][individual];			//current generation
		int32_t newInd[populationSize][individual];		// next generation
		int32_t temp[individual];											// a place to store individual data
		int32_t person1[individual];									// person one
		int32_t person2[individual];									// person two
		float64_t fitness[populationSize];						// the fitness values for the population
		float64_t normFit[populationSize];						// ??
		
		
		
		// methods
		int initPopulation();
		int getFitness(int32_t fitnessFunction, float64_t range);
		int viewPopulation();
		int doCrossover();
		int mutate();
		int swapPopulation();

};

#endif // GACLASS_H

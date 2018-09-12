/*
 * main.cpp
 *
 *  Created on: Jul 26, 2018
 *      Author: john
 */


#include "helper.hpp"
#include "simulator.cuh"

using namespace std;

int main(int argc, char* argv[]) {
	Simulator* sim = new Simulator();
	sim->readTopo("C://Users//jiong.he//source//repos//BGPSimulation//BGPSimulation//data_S.txt");
	sim->alloc_array_val();
	sim->copyToDevice();
	sim->showTopo(-1);

	sim->simulateBGP(1);

	delete sim;
	cout << "Done!" << endl;
	return 0;
}
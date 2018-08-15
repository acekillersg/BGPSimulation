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
	IDR_Simulator* sim = new IDR_Simulator();
	sim->readTopo("C://Users//jiong.he//source//repos//BGPSimulation//BGPSimulation//20170901.as-rel.txt"); // topo file address
	//delete sim;
	cout << "Done!" << endl;
	return 0;
}
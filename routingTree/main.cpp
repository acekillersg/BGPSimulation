//
//  main.cpp
//  IDR_Simulator
//
//  Created by Young on 2/26/18.
//  Copyright © 2018 杨言. All rights reserved.
//

#include <iostream>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include "IDR_Simulator.hpp"
using namespace std;

int main(int argc, const char * argv[])
{
	IDR_Simulator* sim = new IDR_Simulator();
	sim->readTopo("20170901.as-rel.txt"); // topo file address
	sim->simulateBGP(1); // simulate normal BGP process of AS 1
	sim->printASPath1(4); // print the path in routing tree
	sim->simulateHijack(1, 2); // simulate hijacking process launched by AS 2 towards AS 1
	sim->printASPath2(4); // print the path in routing forest
	return 0;
}
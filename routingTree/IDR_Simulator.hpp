//
//  IDR_Simulator.hpp
//  IDR_Simulator
//
//  Created by Young on 2/26/18.
//  Copyright © 2018 杨言. All rights reserved.
//

#ifndef IDR_Simulator_hpp
#define IDR_Simulator_hpp

#define MAX_CONTENT 500000

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include <ctime>
#include <string.h>
#include "Tree.hpp"

using namespace std;

class IDR_Simulator {
public:
	int origin;
	int hijacker;
	int triple;
	int totalAS;
	RouteTree* rt;
	RouteTree* rt_h;
	map<pair<int, int>, int> bottlenecks;
	int* pathLength;
	bool* isHijack;
	vector<int> buffer;
	vector<int>* provider;
	vector<int>* peer;
	vector<int>* customer;
	bool isDigit(string);
	void initSimulation(int, int, int, vector<int>);
	void upstreamSearch(int);
	void oneStepSearch(int);
	void downstreamSearch(int);
	bool pathComp(int, int);
public:
	IDR_Simulator() {
		totalAS = 0;
		rt = NULL;
		rt_h = NULL;
		pathLength = new int[MAX_CONTENT];
		origin = hijacker = triple = -1;
		isHijack = new bool[MAX_CONTENT];
		this->provider = new vector<int>[MAX_CONTENT];
		this->peer = new vector<int>[MAX_CONTENT];
		this->customer = new vector<int>[MAX_CONTENT];
	}
	~IDR_Simulator() {
		if (rt != NULL)
			delete rt;
		if (rt_h != NULL)
			delete rt_h;
		delete[] isHijack;
		delete[] pathLength;
		delete[] this->provider;
		delete[] this->peer;
		delete[] this->customer;
	}
	void showTopo();// for debug only
	void printASPath1(int asn) { rt->printPathToRoot(asn); } // for debug only
	void printASPath2(int asn) { rt_h->printPathToRoot(asn); } // for debug only
	void printTree(ofstream& fout) { rt->printTreeInfo(fout); }
	int getRoutes(int asn1, int asn2) { return rt->getRoutesOnTheLink(asn1, asn2); } // for debug only
	bool readTopo(string);
	void expandTopo(double);
	bool isInInternet(int asn) { return this->provider[asn].size() != 0 || this->peer[asn].size() != 0 || this->customer[asn].size() != 0; }
	bool isInInternet(int, int);
	void simulateBGP(int);
	void simulateHijack(int, int);
	void simulateTripleAnycast(int, int, int);
	void simulateStrategyHijack(int, int, vector<int>);
	double getHijackedRatio();
	int increasedRoutesOnLink(int, int);
	bool printBottlenecks(double, ofstream&, bool anycast = false);
};

#endif /* IDR_Simulator_hpp */
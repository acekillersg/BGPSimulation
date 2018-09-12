 /*
  * Simulator.hpp
  *
  *  Created on: Jul 27, 2018
  *      Author: john
  */

#ifndef SIMULATOR_HPP_
#define SIMULATOR_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include <string.h>
#include "tree.hpp"

using namespace std;

class Simulator {
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

//    vector<int> buffer;
	int* buffer;
    vector<int>* provider;
    vector<int>* peer;
    vector<int>* customer;

	ArrayNode* provider_idx;
	ArrayNode* peer_idx;
	ArrayNode* customer_idx;
	int* provider_val;
	int* peer_val;
	int* customer_val;
	int provider_val_len;
	int peer_val_len;
	int customer_val_len;

	bool isDigit(string);
	void initSimulation(int, int, int, vector<int>);
	void upstreamSearch(int);
	void oneStepSearch(int);
	void downstreamSearch(int);
	bool pathComp(int, int);
public:
	Simulator() {
		totalAS = 0;
		rt = NULL;
		rt_h = NULL;

		//pathLength = new int[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&pathLength, sizeof(int) * MAX_CONTENT));

		origin = hijacker = triple = -1;
		
		//isHijack = new bool[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&isHijack, sizeof(bool) * MAX_CONTENT));

		provider_val_len = 0;
		peer_val_len = 0;
		customer_val_len = 0;

		CudaSafeCall(cudaMallocManaged(&buffer, sizeof(int) * (MAX_CONTENT + 1)));

		this->provider = new vector<int>[MAX_CONTENT];
		this->peer = new vector<int>[MAX_CONTENT];
		this->customer = new vector<int>[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&this->provider_idx, sizeof(ArrayNode) * MAX_CONTENT));
		CudaSafeCall(cudaMallocManaged(&this->peer_idx, sizeof(ArrayNode) * MAX_CONTENT));
		CudaSafeCall(cudaMallocManaged(&this->customer_idx, sizeof(ArrayNode) * MAX_CONTENT));

		CudaSafeCall(cudaMemset(this->provider_idx, 0, sizeof(ArrayNode) * MAX_CONTENT));
		CudaSafeCall(cudaMemset(this->peer_idx, 0, sizeof(ArrayNode) * MAX_CONTENT));
		CudaSafeCall(cudaMemset(this->customer_idx, 0, sizeof(ArrayNode) * MAX_CONTENT));
	}
	~Simulator() {
		if (rt != NULL)
			delete rt;
		if (rt_h != NULL)
			delete rt_h;

		CudaSafeCall(cudaFree(this->pathLength));
		CudaSafeCall(cudaFree(this->isHijack));
		CudaSafeCall(cudaFree(this->buffer));


		delete[] this->provider;
		delete[] this->peer;
		delete[] this->customer;

		CudaSafeCall(cudaFree(this->provider_idx));
		CudaSafeCall(cudaFree(this->peer_idx));
		CudaSafeCall(cudaFree(this->customer_idx));
		CudaSafeCall(cudaFree(this->provider_val));
		CudaSafeCall(cudaFree(this->peer_val));
		CudaSafeCall(cudaFree(this->customer_val));
	}

	void alloc_array_val();
	void copyToDevice();

	void showTopo(int no_to_show);// for debug only
	void printASPath1(int asn); // for debug only
	void printASPath2(int asn); // for debug only
	void printTree(ofstream& fout);
	int getRoutes(int asn1, int asn2); // for debug only
	bool readTopo(string);
	void expandTopo(double);
	bool isInInternet(int asn);
	bool isInInternet(int, int);
	void simulateBGP(int);
	void simulateHijack(int, int);
	void simulateTripleAnycast(int, int, int);
	void simulateStrategyHijack(int, int, vector<int>);
	double getHijackedRatio();
	int increasedRoutesOnLink(int, int);
	bool printBottlenecks(double, ofstream&, bool anycast = false);
};


#endif /* SIMULATOR_HPP_ */
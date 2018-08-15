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


class IDR_Simulator {
private:
	int origin;
	int hijacker;
	int triple;
	int totalAS;
	//RouteTree* rt;
	//RouteTree* rt_h;
	map<pair<int, int>, int> bottlenecks;
	int* pathLength;
	bool* isHijack;

	// changed to GPU
//    vector<int> buffer;
//    vector<int>* provider;
//    vector<int>* peer;
//    vector<int>* customer;
	ArrayNodeHead* provider;
	ArrayNodeHead* peer;
	ArrayNodeHead* customer;


	bool isDigit(string);
	void initSimulation(int, int, int, vector<int>);
	void upstreamSearch(int);
	void oneStepSearch(int);
	void downstreamSearch(int);
	bool pathComp(int, int);
public:
	IDR_Simulator() {
		totalAS = 0;
		//rt = NULL;
		//rt_h = NULL;
		pathLength = new int[MAX_CONTENT];
		origin = hijacker = triple = -1;
		isHijack = new bool[MAX_CONTENT];

		//        this->provider = new vector<int>[MAX_CONTENT];
		//        this->peer = new vector<int>[MAX_CONTENT];
		//        this->customer = new vector<int>[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&this->provider, sizeof(ArrayNodeHead) * MAX_CONTENT));
		CudaSafeCall(cudaMallocManaged(&this->peer, sizeof(ArrayNodeHead) * MAX_CONTENT));
		CudaSafeCall(cudaMallocManaged(&this->customer, sizeof(ArrayNodeHead) * MAX_CONTENT));
	}
	~IDR_Simulator() {
		//if (rt != NULL)
		//	delete rt;
		//if (rt_h != NULL)
		//	delete rt_h;
		delete[] isHijack;
		delete[] pathLength;


		//        delete[] this->provider;
		//        delete[] this->peer;
		//        delete[] this->customer;
		int i;
		for (i = 0; i < MAX_CONTENT; i++) {
			ArrayNode* curr_ptr = this->provider[i].headPtr;
			ArrayNode* next_ptr = nullptr;
			while (curr_ptr != nullptr) {
				next_ptr = curr_ptr->next;
				delete curr_ptr;
				curr_ptr = next_ptr;
			}
			CudaSafeCall(cudaFree(this->provider));
		}
		for (i = 0; i < MAX_CONTENT; i++) {
			ArrayNode* curr_ptr = this->peer[i].headPtr;
			ArrayNode* next_ptr = nullptr;
			while (curr_ptr->next != nullptr) {
				next_ptr = curr_ptr->next;
				delete curr_ptr;
				curr_ptr = next_ptr;
			}
			CudaSafeCall(cudaFree(this->peer));
		}
		for (i = 0; i < MAX_CONTENT; i++) {
			ArrayNode* curr_ptr = this->customer[i].headPtr;
			ArrayNode* next_ptr = nullptr;
			while (curr_ptr->next != nullptr) {
				next_ptr = curr_ptr->next;
				delete curr_ptr;
				curr_ptr = next_ptr;
			}
			CudaSafeCall(cudaFree(this->customer));
		}
	}
	void showTopo();// for debug only
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
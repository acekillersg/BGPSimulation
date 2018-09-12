/*
 * simulator.cu
 *
 *  Created on: Aug 14, 2018
 *      Author: john
 */


#include "simulator.cuh"

bool Simulator::isDigit(string s)
{
	if (s == "")
		return false;
	bool result = true;
	for (size_t i = 0; i < s.size(); i++)
	{
		if (s[i] < '0' || s[i] > '9')
			result = false;
	}
	return result;
}

bool Simulator::readTopo(string filename) {
	try {
		char fbuffer[512];
		ifstream fin(filename);
		if (fin.is_open())
		{
			while (!fin.eof())
			{
				fin.getline(fbuffer, 500);
				if (strlen(fbuffer) > 0 && fbuffer[0] != '#')
				{
					int first_idx = -1;
					int last_idx = -1;
					for (unsigned int i = 0; i < strlen(fbuffer); i++)
					{
						if (fbuffer[i] == '|')
						{
							if (first_idx == -1)
								first_idx = i;
							else
								last_idx = i;
						}
					}
					string sbuffer = fbuffer;
					string asn1 = sbuffer.substr(0, first_idx - 0);
					string asn2 = sbuffer.substr(first_idx + 1, last_idx - first_idx - 1);
					string relation = sbuffer.substr(last_idx + 1, strlen(fbuffer) - last_idx - 1);
					if (isDigit(asn1) && isDigit(asn2))
					{
						int a1 = stoi(asn1);
						int a2 = stoi(asn2);
						if (relation == "0")
						{
							this->peer[a1].push_back(a2);
							this->peer[a2].push_back(a1);
						}
						if (relation == "-1")
						{
							this->customer[a1].push_back(a2);
							this->provider[a2].push_back(a1);
						}
					}
					else
						throw string("Error:TopoDataFormat");
				}
			}
			fin.close();
			for (int i = 0; i < MAX_CONTENT; i++)
			{
				if (this->provider[i].size() != 0 || this->peer[i].size() != 0 || this->customer[i].size() != 0)
					this->totalAS++;
			}

			// get the length of three arrays
			for (int i = 0; i < MAX_CONTENT; i++) {
				this->peer_val_len += this->peer[i].size();
				this->customer_val_len += this->customer[i].size();
				this->provider_val_len += this->provider[i].size();
			}
			return true;
		}
		else
			throw string("Error:TopoFileOpening");
	}
	catch (exception &err) {
		cout << err.what() << endl;
		return false;
	}
	catch (string &myerror) {
		cout << myerror << endl;
		return false;
	}
}

void Simulator::alloc_array_val() {
	CudaSafeCall(cudaMallocManaged(&this->provider_val, sizeof(int) * this->provider_val_len));
	CudaSafeCall(cudaMallocManaged(&this->peer_val, sizeof(int) * this->peer_val_len));
	CudaSafeCall(cudaMallocManaged(&this->customer_val, sizeof(int) * this->customer_val_len));

	CudaSafeCall(cudaMemset(this->provider_val, 0, sizeof(int) * this->provider_val_len));
	CudaSafeCall(cudaMemset(this->peer_val, 0, sizeof(int) * this->peer_val_len));
	CudaSafeCall(cudaMemset(this->customer_val, 0, sizeof(int) * this->customer_val_len));
}

void Simulator::copyToDevice() {

	int peer_ptr = 0, customer_ptr = 0, provider_ptr = 0;
	for (int i = 0; i < MAX_CONTENT; i++) {

		// deep copy of peer
		if (this->peer[i].size() != 0) {
			this->peer_idx[i].start_idx = peer_ptr;
			for (int j = 0; j < this->peer[i].size(); j++) {
				this->peer_val[peer_ptr + j] = this->peer[i].at(j);
			}
			peer_ptr += this->peer[i].size();
			this->peer_idx[i].end_idx = peer_ptr;
		}
		else {
			this->peer_idx[i].start_idx = peer_ptr;
			this->peer_idx[i].end_idx = peer_ptr;
		}

		// deep copy of customer
		if (this->customer[i].size() != 0) {
			this->customer_idx[i].start_idx = customer_ptr;
			for (int j = 0; j < this->customer[i].size(); j++) {
				this->customer_val[customer_ptr + j] = this->customer[i].at(j);
			}
			customer_ptr += this->customer[i].size();
			this->customer_idx[i].end_idx = customer_ptr;
		}
		else {
			this->customer_idx[i].start_idx = customer_ptr;
			this->customer_idx[i].end_idx = customer_ptr;
		}

		// deep copy of provider
		if (this->provider[i].size() != 0) {
			this->provider_idx[i].start_idx = provider_ptr;
			for (int j = 0; j < this->provider[i].size(); j++) {
				this->provider_val[provider_ptr + j] = this->provider[i].at(j);
			}
			provider_ptr += this->provider[i].size();
			this->provider_idx[i].end_idx = provider_ptr;
		}
		else {
			this->provider_idx[i].start_idx = provider_ptr;
			this->provider_idx[i].end_idx = provider_ptr;
		}
	}
}

void Simulator::showTopo(int no_to_show)
{
	if (no_to_show == -1) no_to_show = MAX_CONTENT;
	for (int i = 0; i < no_to_show; i++)
	{
		int start_idx = this->provider_idx[i].start_idx;
		int end_idx = this->provider_idx[i].end_idx;
		if (start_idx != end_idx) {
			for (int j = start_idx; j < end_idx; j++)
				cout << "provider of " << i << " is " << this->provider_val[j] << endl;
		}
	}
	for (int i = 0; i < no_to_show; i++)
	{
		int start_idx = this->customer_idx[i].start_idx;
		int end_idx = this->customer_idx[i].end_idx;
		if (start_idx != end_idx) {
			for (int j = start_idx; j < end_idx; j++)
				cout << "customer of " << i << " is " << this->customer_val[j] << endl;
		}
	}
	for (int i = 0; i < no_to_show; i++)
	{
		int start_idx = this->peer_idx[i].start_idx;
		int end_idx = this->peer_idx[i].end_idx;
		if (start_idx != end_idx) {
			for (int j = start_idx; j < end_idx; j++)
				cout << "peer of " << i << " is " << this->peer_val[j] << endl;
		}
	}
}


void Simulator::initSimulation(int asn1, int asn2 = -1, int asn3 = -1, vector<int> p = vector<int>()) {
	if (asn2 == -1) {
		if (this->rt != NULL)
			delete this->rt;
		if (this->rt_h != NULL)
			delete this->rt_h;

		this->rt = new RouteTree(asn1);
		
		this->origin = asn1;
		//this->buffer.clear();
		//this->buffer.push_back(asn1);
		for (int i = 0; i < MAX_CONTENT; i++)
			pathLength[i] = -1;
		pathLength[asn1] = 0;
	}  
	else {

	}
}

void Simulator::upstreamSearch(int treeNo) {

}

void Simulator::oneStepSearch(int treeNo) {

}

void Simulator::downstreamSearch(int treeNo) {

}



void Simulator::simulateBGP(int asn)
{
	// cout << "simulate BGP process of AS " << asn << endl;
	initSimulation(asn);

	//upstreamSearch(1);

	//oneStepSearch(1);

	//downstreamSearch(1);
}
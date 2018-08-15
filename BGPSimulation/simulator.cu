/*
 * simulator.cu
 *
 *  Created on: Aug 14, 2018
 *      Author: john
 */


#include "simulator.cuh"

bool IDR_Simulator::isDigit(string s)
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

bool IDR_Simulator::readTopo(string filename) {
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
							//this->peer[a1].push_back(a2);
							//this->peer[a2].push_back(a1);
							if (this->peer[a1].headPtr == nullptr) {
								this->peer[a1].headPtr = new ArrayNode(a2);
								this->peer[a1].tailPtr = this->peer[a1].headPtr;
							}
							else {
								this->peer[a1].tailPtr->next = new ArrayNode(a2);
								this->peer[a1].tailPtr = this->peer[a1].tailPtr->next;
							}

							if (this->peer[a2].headPtr == nullptr) {
								this->peer[a2].headPtr = new ArrayNode(a1);
								this->peer[a2].tailPtr = this->peer[a2].headPtr;
							}
							else {
								this->peer[a2].tailPtr->next = new ArrayNode(a1);
								this->peer[a2].tailPtr = this->peer[a2].tailPtr->next;
							}
						}
						if (relation == "-1")
						{
							//this->customer[a1].push_back(a2);
							//this->provider[a2].push_back(a1);
							if (this->customer[a1].headPtr == nullptr) {
								this->customer[a1].headPtr = new ArrayNode(a2);
								this->customer[a1].tailPtr = this->customer[a1].headPtr;
							}
							else {
								this->customer[a1].tailPtr->next = new ArrayNode(a2);
								this->customer[a1].tailPtr = this->customer[a1].tailPtr->next;
							}

							if (this->provider[a2].headPtr == nullptr) {
								this->provider[a2].headPtr = new ArrayNode(a1);
								this->provider[a2].tailPtr = this->provider[a2].headPtr;
							}
							else {
								this->provider[a2].tailPtr->next = new ArrayNode(a1);
								this->provider[a2].tailPtr = this->provider[a2].tailPtr->next;
							}
						}
					}
					else
						throw string("Error:TopoDataFormat");
				}
			}
			fin.close();
			for (int i = 0; i < MAX_CONTENT; i++)
			{
				//if (this->provider[i].size() != 0 || this->peer[i].size() != 0 || this->customer[i].size() != 0)
					//this->totalAS++;
				if (this->provider[i].headPtr != nullptr || this->peer[i].headPtr != nullptr || this->customer[i].headPtr != nullptr)
					this->totalAS++;
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
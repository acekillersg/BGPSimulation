//
//  IDR_Simulator.cpp
//  IDR_Simulator
//
//  Created by Young on 2/26/18.
//  Copyright © 2018 杨言. All rights reserved.
//
#include "IDR_Simulator.hpp"
bool reverseComp(int i, int j) { return (i > j); }
bool mapComp(const pair<pair<int, int>, int> x, const pair<pair<int, int>, int> y) { return x.second > y.second; }
bool IDR_Simulator::pathComp(int i, int j) { return ((pathLength[i] > pathLength[j]) || (pathLength[i] == pathLength[j] && i > j)); }

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

void IDR_Simulator::initSimulation(int asn1, int asn2 = -1, int asn3 = -1, vector<int> p = vector<int>())
{
	if (asn2 == -1)
	{
		if (rt != NULL)
			delete rt;
		if (rt_h != NULL)
			delete rt_h;
		rt = new RouteTree(asn1);
		this->origin = asn1;
		this->buffer.clear();
		this->buffer.push_back(asn1);
		for (int i = 0; i < MAX_CONTENT; i++)
			pathLength[i] = -1;
		pathLength[asn1] = 0;
	}
	else
	{
		if (p.size() == 0)
		{
			if (rt_h != NULL)
				delete rt_h;
			if (asn3 == -1)
				rt_h = new RouteTree(asn1, asn2);
			else
				rt_h = new RouteTree(asn1, asn2, asn3);
			this->origin = asn1;
			this->hijacker = asn2;
			this->buffer.clear();
			this->buffer.push_back(asn1);
			this->buffer.push_back(asn2);
			for (int i = 0; i < MAX_CONTENT; i++)
			{
				pathLength[i] = -1;
				isHijack[i] = false;
			}
			pathLength[asn1] = pathLength[asn2] = 0;
			if (asn3 != -1)
			{
				this->triple = asn3;
				this->buffer.push_back(asn3);
				pathLength[asn3] = 0;
			}
			isHijack[asn2] = true;
		}
		else
		{
			if (rt_h != NULL)
				delete rt_h;
			rt_h = new RouteTree(asn1, asn2);
			this->origin = asn1;
			this->hijacker = asn2;
			this->buffer.clear();
			this->buffer.push_back(asn1);
			this->buffer.push_back(asn2);
			for (int i = 0; i < MAX_CONTENT; i++)
			{
				pathLength[i] = -1;
				isHijack[i] = false;
			}
			for (int i = 0; i < (int)p.size(); i++)
			{
				pathLength[p[i]] = -2;
			}
			pathLength[asn1] = 0;
			pathLength[asn2] = (int)p.size() + 1;
			isHijack[asn2] = true;
		}
	}
}

void IDR_Simulator::upstreamSearch(int treeNo)
{
	RouteTree* thert;
	if (treeNo == 1)
		thert = rt;
	else
		thert = rt_h;
	vector<int> active = this->buffer;
	this->buffer.clear();
	auto bound_comp = bind(&IDR_Simulator::pathComp, this, std::placeholders::_1, std::placeholders::_2);
	sort(active.begin(), active.end(), bound_comp);
	int currentPathLength = 0;
	vector<int> candidate;
	while (active.size() > 0)
	{
		int current = *(active.end() - 1);
		active.pop_back();
		this->buffer.insert(this->buffer.begin(), current);
		currentPathLength = this->pathLength[current];
		if (this->provider[current].size() != 0)
		{
			for (unsigned int i = 0; i < this->provider[current].size(); i++)
			{
				if (thert->addNode(current, this->provider[current][i]))
				{
					if (isHijack[current])
					{
						if (pathLength[this->provider[current][i]] == -2)
						{
							continue;
						}
						else
						{
							candidate.push_back(this->provider[current][i]);
							this->pathLength[this->provider[current][i]] = this->pathLength[current] + 1;
							isHijack[this->provider[current][i]] = true;
						}
					}
					else
					{
						candidate.push_back(this->provider[current][i]);
						this->pathLength[this->provider[current][i]] = this->pathLength[current] + 1;
					}
				}
			}
		}
		if (active.size() == 0 || currentPathLength != this->pathLength[active[active.size() - 1]])
		{
			auto bound_comp = bind(&IDR_Simulator::pathComp, this, std::placeholders::_1, std::placeholders::_2);
			while (candidate.size() > 0)
			{
				int c = *(candidate.end() - 1);
				candidate.pop_back();
				active.push_back(c);
			}
			sort(active.begin(), active.end(), bound_comp);
		}
	}
}

void IDR_Simulator::oneStepSearch(int treeNo)
{
	RouteTree* thert;
	if (treeNo == 1)
		thert = rt;
	else
		thert = rt_h;
	vector<int> active = this->buffer;
	vector<int> candidate;
	while (active.size() > 0)
	{
		int current = *(active.end() - 1);
		active.pop_back();
		if (this->peer[current].size() != 0)
		{
			for (unsigned int i = 0; i < this->peer[current].size(); i++)
			{
				if (thert->addNode(current, this->peer[current][i]))
				{
					if (isHijack[current])
					{
						if (pathLength[this->peer[current][i]] == -2)
						{
							continue;
						}
						else
						{
							candidate.push_back(this->peer[current][i]);
							this->pathLength[this->peer[current][i]] = this->pathLength[current] + 1;
							isHijack[this->peer[current][i]] = true;
						}
					}
					else
					{
						candidate.push_back(this->peer[current][i]);
						this->pathLength[this->peer[current][i]] = this->pathLength[current] + 1;
					}
				}
			}
		}
	}
	active = candidate;
	while (active.size() > 0)
	{
		int current = *(active.end() - 1);
		active.pop_back();
		this->buffer.insert(this->buffer.begin(), current);
	}
	auto bound_comp = bind(&IDR_Simulator::pathComp, this, std::placeholders::_1, std::placeholders::_2);
	sort(this->buffer.begin(), this->buffer.end(), bound_comp);
}

void IDR_Simulator::downstreamSearch(int treeNo)
{
	RouteTree* thert;
	if (treeNo == 1)
		thert = rt;
	else
		thert = rt_h;
	vector<int> active = this->buffer;
	this->buffer.clear();
	int currentPathLength = 0;
	vector<int> candidate;
	while (active.size() > 0)
	{
		int current = *(active.end() - 1);
		active.pop_back();
		this->buffer.insert(this->buffer.begin(), current);
		currentPathLength = this->pathLength[current];
		if (this->customer[current].size() != 0)
		{
			for (unsigned int i = 0; i < this->customer[current].size(); i++)
			{
				if (thert->addNode(current, this->customer[current][i]))
				{
					if (isHijack[current])
					{
						if (pathLength[this->customer[current][i]] == -2)
						{
							continue;
						}
						else
						{
							candidate.push_back(this->customer[current][i]);
							this->pathLength[this->customer[current][i]] = this->pathLength[current] + 1;
							isHijack[this->customer[current][i]] = true;
						}
					}
					else
					{
						candidate.push_back(this->customer[current][i]);
						this->pathLength[this->customer[current][i]] = this->pathLength[current] + 1;
					}
				}
			}
		}
		if (active.size() == 0 || currentPathLength != this->pathLength[active[active.size() - 1]])
		{
			auto bound_comp = bind(&IDR_Simulator::pathComp, this, std::placeholders::_1, std::placeholders::_2);
			while (candidate.size() > 0)
			{
				int c = *(candidate.end() - 1);
				candidate.pop_back();
				active.push_back(c);
			}
			sort(active.begin(), active.end(), bound_comp);
		}
	}
}

void IDR_Simulator::showTopo()
{
	for (int i = 0; i < MAX_CONTENT; i++)
	{
		if (this->provider[i].size() != 0)
			for (unsigned int j = 0; j < this->provider[i].size(); j++)
				cout << "provider of " << i << " is " << this->provider[i][j] << endl;
	}
	for (int i = 0; i < MAX_CONTENT; i++)
	{
		if (this->customer[i].size() != 0)
			for (unsigned int j = 0; j < this->customer[i].size(); j++)
				cout << "customer of " << i << " is " << this->customer[i][j] << endl;
	}
	for (int i = 0; i < MAX_CONTENT; i++)
	{
		if (this->peer[i].size() != 0)
			for (unsigned int j = 0; j < this->peer[i].size(); j++)
				cout << "peer of " << i << " is " << this->peer[i][j] << endl;
	}
}

bool IDR_Simulator::readTopo(string filename)
{
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
			return true;
		}
		else
			throw string("Error:TopoFileOpening");
	}
	catch (exception &err) {
		cout << err.what() << endl;
		return false;
	}
	catch (string& myerror) {
		cout << myerror << endl;
		return false;
	}
}

void IDR_Simulator::expandTopo(double prop)
{
	int cnt_peer = 0;
	for (int i = 0; i < MAX_CONTENT; i++)
	{
		cnt_peer += (int)this->peer[i].size();
	}
	cnt_peer = cnt_peer / 2;
	//cout << cnt_peer << endl;
	int cnt_add = cnt_peer / prop - cnt_peer;
	srand((unsigned int)time(NULL));
	while (cnt_add > 0)
	{
		int asn1 = rand() % MAX_CONTENT;
		int asn2 = rand() % MAX_CONTENT;
		if (asn1 != asn2)
		{
			vector<int>::iterator it = find(this->peer[asn1].begin(), this->peer[asn1].end(), asn2);
			if (it == this->peer[asn1].end())
			{
				this->peer[asn1].push_back(asn2);
				this->peer[asn2].push_back(asn1);
				cnt_add--;
			}
		}
	}
}

void IDR_Simulator::simulateBGP(int asn)
{
	//cout << "simulate BGP process of AS " << asn << endl;
	initSimulation(asn);
	upstreamSearch(1);
	oneStepSearch(1);
	downstreamSearch(1);
}

void IDR_Simulator::simulateHijack(int victim, int attacker)
{
	//cout << "simulate hijacking launched by AS " << attacker << " towards AS " << victim << endl;
	if (victim == attacker)
		return;
	initSimulation(victim, attacker);
	upstreamSearch(2);
	oneStepSearch(2);
	downstreamSearch(2);
}

void IDR_Simulator::simulateStrategyHijack(int victim, int attacker, vector<int> prepending)
{
	if (victim == attacker)
		return;
	initSimulation(victim, attacker, -1, prepending);
	upstreamSearch(2);
	oneStepSearch(2);
	downstreamSearch(2);
}

void IDR_Simulator::simulateTripleAnycast(int member1, int member2, int member3)
{
	if (member1 == member2 || member1 == member3 || member2 == member3)
		return;
	initSimulation(member1, member2, member3);
	upstreamSearch(2);
	oneStepSearch(2);
	downstreamSearch(2);
}

int IDR_Simulator::increasedRoutesOnLink(int asn1, int asn2)
{
	if (rt_h == NULL || rt == NULL)
	{
		cout << "Please simulate BGP process and hijack first!" << endl;
		return 0;
	}
	return rt_h->getRoutesOnTheLink(asn1, asn2) - rt->getRoutesOnTheLink(asn1, asn2);
}

double IDR_Simulator::getHijackedRatio()
{
	if (rt_h == NULL)
		return 0;
	int hijack_cnt = 0;
	for (int i = 0; i < MAX_CONTENT; i++)
	{
		if (this->provider[i].size() != 0 || this->peer[i].size() != 0 || this->customer[i].size() != 0)
		{
			vector<int> path = rt_h->getPathToRoot(i);
			if (path.size() >= 1)
			{
				int source = *(path.end() - 1);
				if (source == hijacker)
					hijack_cnt++;
			}
		}
	}
	return (double)hijack_cnt / totalAS;
}

bool IDR_Simulator::isInInternet(int asn1, int asn2)
{
	vector<int>::iterator it = find(this->provider[asn1].begin(), this->provider[asn1].end(), asn2);
	if (it != this->provider[asn1].end())
		return true;
	it = find(this->peer[asn1].begin(), this->peer[asn1].end(), asn2);
	if (it != this->peer[asn1].end())
		return true;
	it = find(this->customer[asn1].begin(), this->customer[asn1].end(), asn2);
	if (it != this->customer[asn1].end())
		return true;
	return false;
}

bool IDR_Simulator::printBottlenecks(double threshold, ofstream& fout, bool anycast)
{
	if (!anycast && rt == NULL)
	{
		cout << "Please simulate BGP process first!" << endl;
		return false;
	}
	if (anycast && rt_h == NULL)
	{
		cout << "Please simulate Hijack(Anycast) process first!" << endl;
		return false;
	}
	this->bottlenecks.clear();
	if (!anycast)
		this->bottlenecks = rt->getBottlenecks(threshold * (totalAS - 1));
	else
		this->bottlenecks = rt_h->getBottlenecks(threshold * (totalAS - 1), anycast);
	if (this->bottlenecks.size() == 0)
		return false;
	vector<pair<pair<int, int>, int> > sortedNecks;
	for (map<pair<int, int>, int>::iterator cur = this->bottlenecks.begin(); cur != this->bottlenecks.end(); cur++)
	{
		sortedNecks.push_back(make_pair(cur->first, cur->second));
	}
	sort(sortedNecks.begin(), sortedNecks.end(), mapComp);
	vector<pair<pair<int, int>, int> >::iterator it = sortedNecks.begin();
	while (it != sortedNecks.end())
	{
		if (!anycast)
			//fout << this->origin << " " << it->first.first << " " << it->first.second << " " << ((double)it->second)/(totalAS-1) << endl;
			fout << ((double)it->second) / (totalAS - 1) << endl;
		else
		{
			if (this->triple == -1)
				fout << this->origin << " " << this->hijacker << " " << it->first.first << " " << it->first.second << " " << ((double)it->second) / (totalAS - 1) << endl;
			else
				fout << this->origin << " " << this->hijacker << " " << this->triple << " " << it->first.first << " " << it->first.second << " " << ((double)it->second) / (totalAS - 1) << endl;
		}
		it++;
	}
	return true;
}
/*
 * Tree.hpp
 *
 *  Created on: Jul 26, 2018
 *      Author: john
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include "helper.hpp"
#include <fstream>
#include <vector>

using namespace std;


#define MAX_CONTENT 500000

 // define base class to allocate objects in unified memory
class ManagedMemObj {
public:
	void* operator new (size_t len) {
		void* ptr;
		CudaSafeCall(cudaMallocManaged(&ptr, len));
		cudaDeviceSynchronize();
		return ptr;
	}

	void* operator new[](size_t len) {
		void* ptr;
		CudaSafeCall(cudaMallocManaged(&ptr, len));
		cudaDeviceSynchronize();
		return ptr;
	}


		void operator delete (void* ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}

	void operator delete[](void* ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class TreeNode : public ManagedMemObj {
public:
	int depth;
	int element;
	TreeNode* parent;
	TreeNode* firstChild;
	TreeNode* lastSibling;
	TreeNode* nextSibling;

	TreeNode() { depth = 0; element = 0; parent = NULL; firstChild = NULL; lastSibling = NULL; nextSibling = NULL; }
	TreeNode(int asn) { depth = 0; element = asn; parent = NULL; firstChild = NULL; lastSibling = NULL; nextSibling = NULL; }

	// copy constructor, enabling pass-by-value
	TreeNode(const TreeNode& other) {
		depth = other.depth;
		element = other.element;
		parent = other.parent;
		firstChild = other.firstChild;
		lastSibling = other.lastSibling;
		nextSibling = other.nextSibling;
	}

	// overload assignment operator
	TreeNode& operator=(const TreeNode& other) {
		depth = other.depth;
		element = other.element;
		parent = other.parent;
		firstChild = other.firstChild;
		lastSibling = other.lastSibling;
		nextSibling = other.nextSibling;
		return *this;
	}
	~TreeNode() {}
};


class RouteTree : public ManagedMemObj {
private:
	int root1value;
	int root2value;
	int root3value;
	TreeNode* root;
	TreeNode** markInTree;
	int subTreeSize;
	void DFSWholeTree(TreeNode*);
	void printTreeByDeep(TreeNode*, ofstream&);
	bool traversalSubTree(int);
	bool printSubTree(int, ofstream&);
public:
	RouteTree(int originAS)
	{
		root1value = originAS;
		root2value = -1;
		root3value = -1;
		root = new TreeNode(originAS);

		// markInTree = new TreeNode*[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&markInTree, sizeof(TreeNode*) * MAX_CONTENT));

		for (int i = 0; i < MAX_CONTENT; i++)
			markInTree[i] = NULL;

		markInTree[originAS] = root;
		subTreeSize = 0;
	}
	RouteTree(int firstOrigin, int secondOrigin)
	{
		root1value = firstOrigin;
		root2value = secondOrigin;
		root3value = -1;
		root = new TreeNode[2];
		root[0].element = firstOrigin;
		root[1].element = secondOrigin;

		//markInTree = new TreeNode*[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&markInTree, sizeof(TreeNode*) * MAX_CONTENT));

		for (int i = 0; i < MAX_CONTENT; i++)
			markInTree[i] = NULL;

		markInTree[firstOrigin] = &root[0];
		markInTree[secondOrigin] = &root[1];
		subTreeSize = 0;
	}
	RouteTree(int firstOrigin, int secondOrigin, int thirdOrigin)
	{
		root1value = firstOrigin;
		root2value = secondOrigin;
		root3value = thirdOrigin;
		root = new TreeNode[3];
		root[0].element = firstOrigin;
		root[1].element = secondOrigin;
		root[2].element = thirdOrigin;

		//markInTree = new TreeNode*[MAX_CONTENT];
		CudaSafeCall(cudaMallocManaged(&markInTree, sizeof(TreeNode*) * MAX_CONTENT));

		for (int i = 0; i < MAX_CONTENT; i++)
			markInTree[i] = NULL;

		markInTree[firstOrigin] = &root[0];
		markInTree[secondOrigin] = &root[1];
		markInTree[thirdOrigin] = &root[2];
		subTreeSize = 0;
	}
	~RouteTree()
	{
		if (root2value != -1)
		{
			//delete[] markInTree[root1value];
			//markInTree[root1value] = NULL;
			//markInTree[root2value] = NULL;
			//if (root3value != -1)
			//	markInTree[root3value] = NULL;
		}
		for (int i = 0; i < MAX_CONTENT; i++)
		{
			//if (markInTree[i] != NULL)
			//	delete[] markInTree[i];
		}
		CudaSafeCall(cudaFree(markInTree));
	}
	bool addNode(int, int);
	bool printPathToRoot(int);
	void printTreeInfo(ofstream&);
	vector<int> getPathToRoot(int);
	int getRoutesOnTheLink(int, int);
	//map<pair<int, int>, int> getBottlenecks(int, bool anycast = false);
};

class ArrayNode {
public:
	int start_idx;
	int end_idx;

	ArrayNode() : start_idx(0), end_idx(0) {}
	~ArrayNode() {}
};

#endif /* TREE_HPP_ */
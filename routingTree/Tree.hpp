//
//  Tree.hpp
//  IDR_Simulator
//
//  Created by Young on 2/26/18.
//  Copyright © 2018 杨言. All rights reserved.
//

#ifndef Tree_hpp
#define Tree_hpp

#define MAX_CONTENT 500000

#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

struct TreeNode {
	int depth;
	int element;
	TreeNode* parent;
	TreeNode* firstChild;
	TreeNode* lastSibling;
	TreeNode* nextSibling;
	TreeNode() { depth = 0; element = 0; parent = NULL; firstChild = NULL; lastSibling = NULL; nextSibling = NULL; }
	TreeNode(int asn) { depth = 0; element = asn; parent = NULL; firstChild = NULL; lastSibling = NULL; nextSibling = NULL; }
	~TreeNode() {}
};

class RouteTree {
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
		markInTree = new TreeNode*[MAX_CONTENT];
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
		markInTree = new TreeNode*[MAX_CONTENT];
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
		markInTree = new TreeNode*[MAX_CONTENT];
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
			delete[] markInTree[root1value];
			markInTree[root1value] = NULL;
			markInTree[root2value] = NULL;
			if (root3value != -1)
				markInTree[root3value] = NULL;
		}
		for (int i = 0; i < MAX_CONTENT; i++)
		{
			if (markInTree[i] != NULL)
				delete markInTree[i];
		}
		delete[] markInTree;
	}
	bool addNode(int, int);
	bool printPathToRoot(int);
	void printTreeInfo(ofstream&);
	vector<int> getPathToRoot(int);
	int getRoutesOnTheLink(int, int);
	map<pair<int, int>, int> getBottlenecks(int, bool anycast = false);
};

#endif /* Tree_hpp */
//
//  Tree.cpp
//  IDR_Simulator
//
//  Created by Young on 2/26/18.
//  Copyright © 2018 杨言. All rights reserved.
//

#include "Tree.hpp"

void RouteTree::DFSWholeTree(TreeNode* node)
{
	subTreeSize++;
	if (node->firstChild != NULL)
	{
		node = node->firstChild;
		DFSWholeTree(node);
		while (node->nextSibling != NULL)
		{
			node = node->nextSibling;
			DFSWholeTree(node);
		}
	}
}

void RouteTree::printTreeByDeep(TreeNode * node, ofstream& fout)
{
	if (node->firstChild != NULL)
	{
		fout << (node->element) << " " << (node->firstChild->element) << endl;
		node = node->firstChild;
		printTreeByDeep(node, fout);
		while (node->nextSibling != NULL)
		{
			fout << (node->parent->element) << " " << (node->nextSibling->element) << endl;
			node = node->nextSibling;
			printTreeByDeep(node, fout);
		}
	}
}

bool RouteTree::traversalSubTree(int asn)
{
	if (markInTree[asn] == 0)
		return false;
	subTreeSize = 0;
	TreeNode* current = markInTree[asn];
	DFSWholeTree(current);
	return true;
}

bool RouteTree::printSubTree(int asn, ofstream& fout)
{
	if (markInTree[asn] == 0)
		return false;
	TreeNode* current = markInTree[asn];
	printTreeByDeep(current, fout);
	return true;
}

bool RouteTree::addNode(int p, int c)
{
	if (markInTree[p] == NULL || markInTree[c] != NULL)
		return false;
	TreeNode* node = new TreeNode(c);
	node->parent = markInTree[p];
	node->depth = markInTree[p]->depth + 1;
	if (markInTree[p]->firstChild == NULL)
		markInTree[p]->firstChild = node;
	else
	{
		markInTree[p]->firstChild->lastSibling = node;
		node->nextSibling = markInTree[p]->firstChild;
		markInTree[p]->firstChild = node;
	}
	markInTree[c] = node;
	return true;
}

bool RouteTree::printPathToRoot(int asn)
{
	if (markInTree[asn] == 0)
		return false;
	TreeNode* current = markInTree[asn];
	cout << current->element << " ";
	while (current->parent != NULL)
	{
		current = current->parent;
		cout << current->element << " ";
	}
	cout << endl;
	return true;
}

vector<int> RouteTree::getPathToRoot(int asn)
{
	vector<int> result;
	if (markInTree[asn] == 0)
		return result;
	TreeNode* current = markInTree[asn];
	result.push_back(current->element);
	while (current->parent != NULL)
	{
		current = current->parent;
		result.push_back(current->element);
	}
	return result;
}

int RouteTree::getRoutesOnTheLink(int asn1, int asn2)
{
	if ((markInTree[asn1] == NULL) || (markInTree[asn2] == NULL) || (markInTree[asn1]->parent != markInTree[asn2] && markInTree[asn2]->parent != markInTree[asn1]))
		return 0;
	int count = 0;
	if (markInTree[asn1]->parent == markInTree[asn2])
	{
		traversalSubTree(asn1);
		count = subTreeSize;
	}
	else
	{
		traversalSubTree(asn2);
		count = subTreeSize;
	}
	return count;
}

map<pair<int, int>, int> RouteTree::getBottlenecks(int threshold, bool anycast)
{
	map<pair<int, int>, int> bn;
	if (this->root->firstChild != NULL)
	{
		TreeNode* node = this->root->firstChild;
		vector<int> active;
		active.push_back(node->element);
		while (node->nextSibling != NULL)
		{
			node = node->nextSibling;
			active.push_back(node->element);
		}
		while (active.size() > 0)
		{
			vector<int> candidate;
			while (active.size() > 0)
			{
				int current = active[active.size() - 1];
				active.pop_back();
				traversalSubTree(current);
				if (subTreeSize >= threshold)
				{
					TreeNode* n = markInTree[current];
					int asn1 = current;
					int asn2 = n->parent->element;
					if (asn1 >= asn2)
					{
						int tmp = asn1;
						asn1 = asn2;
						asn2 = tmp;
					}
					bn[make_pair(asn1, asn2)] = subTreeSize;
					if (n->firstChild != NULL)
					{
						n = n->firstChild;
						candidate.push_back(n->element);
						while (n->nextSibling != NULL)
						{
							n = n->nextSibling;
							candidate.push_back(n->element);
						}
					}
				}
			}
			active = candidate;
		}
	}
	if (anycast)
	{
		if (this->root[1].firstChild != NULL)
		{
			TreeNode* node = this->root[1].firstChild;
			vector<int> active;
			active.push_back(node->element);
			while (node->nextSibling != NULL)
			{
				node = node->nextSibling;
				active.push_back(node->element);
			}
			while (active.size() > 0)
			{
				vector<int> candidate;
				while (active.size() > 0)
				{
					int current = active[active.size() - 1];
					active.pop_back();
					traversalSubTree(current);
					if (subTreeSize >= threshold)
					{
						TreeNode* n = markInTree[current];
						int asn1 = current;
						int asn2 = n->parent->element;
						if (asn1 >= asn2)
						{
							int tmp = asn1;
							asn1 = asn2;
							asn2 = tmp;
						}
						bn[make_pair(asn1, asn2)] = subTreeSize;
						if (n->firstChild != NULL)
						{
							n = n->firstChild;
							candidate.push_back(n->element);
							while (n->nextSibling != NULL)
							{
								n = n->nextSibling;
								candidate.push_back(n->element);
							}
						}
					}
				}
				active = candidate;
			}
		}
		if (this->root3value != -1)
		{
			if (this->root[2].firstChild != NULL)
			{
				TreeNode* node = this->root[2].firstChild;
				vector<int> active;
				active.push_back(node->element);
				while (node->nextSibling != NULL)
				{
					node = node->nextSibling;
					active.push_back(node->element);
				}
				while (active.size() > 0)
				{
					vector<int> candidate;
					while (active.size() > 0)
					{
						int current = active[active.size() - 1];
						active.pop_back();
						traversalSubTree(current);
						if (subTreeSize >= threshold)
						{
							TreeNode* n = markInTree[current];
							int asn1 = current;
							int asn2 = n->parent->element;
							if (asn1 >= asn2)
							{
								int tmp = asn1;
								asn1 = asn2;
								asn2 = tmp;
							}
							bn[make_pair(asn1, asn2)] = subTreeSize;
							if (n->firstChild != NULL)
							{
								n = n->firstChild;
								candidate.push_back(n->element);
								while (n->nextSibling != NULL)
								{
									n = n->nextSibling;
									candidate.push_back(n->element);
								}
							}
						}
					}
					active = candidate;
				}
			}
		}
	}
	return bn;
}

void RouteTree::printTreeInfo(ofstream& fout)
{
	if (root2value == -1)
	{
		printSubTree(root->element, fout);
	}
	else
	{
		printSubTree(root[0].element, fout);
		printSubTree(root[1].element, fout);
		if (root3value != -1)
			printSubTree(root[2].element, fout);
	}
}
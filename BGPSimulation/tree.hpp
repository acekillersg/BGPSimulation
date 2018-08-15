/*
 * Tree.hpp
 *
 *  Created on: Jul 26, 2018
 *      Author: john
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include "helper.hpp"

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

class ArrayNode : public ManagedMemObj {
public:
	int val;
	ArrayNode* next;

	ArrayNode() : val(0), next(nullptr) {}
	ArrayNode(int v) {
		this->val = v;
		this->next = nullptr;
	}

	// assignment function overloading
	ArrayNode& operator= (const ArrayNode& other) {
		this->val = other.val;
		this->next = other.next;
		return *this;
	}

	~ArrayNode() {}
};

class ArrayNodeHead : public ManagedMemObj {
public:
	int AS;
	ArrayNode* headPtr;
	ArrayNode* tailPtr;

	ArrayNodeHead() : AS(-1), headPtr(nullptr), tailPtr(nullptr) {}
	~ArrayNodeHead() {}
};

#endif /* TREE_HPP_ */
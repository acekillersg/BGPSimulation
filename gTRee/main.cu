#include "helper.hpp"

#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;



class ManagedMem {
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


class DataElem : public ManagedMem {
public:
	int prop;

	// constructor function overloading
	DataElem() : prop(0) {}

	// copy constructor overloading
	DataElem(const DataElem &s) {
		this->prop = s.prop;
	}

	// assignment function overloading
	DataElem& operator= (const DataElem& other) {
		this->prop = other.prop;
		return *this;
	}

	// destructor function overloading
	~DataElem() {}
};



// Kernel function to add the elements of two arrays
__global__
void add(DataElem** data) {
	printf("On device, data[0][0]: value = %d\n", data[0][0].prop);
	printf("On device, data[0][1]: value = %d\n", data[0][1].prop);
	printf("On device, data[1][0]: value = %d\n", data[1][0].prop);
	printf("On device, data[1][1]: value = %d\n", data[1][1].prop);
	printf("On device, data[1][2]: value = %d\n", data[1][2].prop);

	data[0][0].prop += 10;
	data[0][1].prop += 10;
	data[1][0].prop += 10;
	data[1][1].prop += 10;
	data[1][2].prop += 10;
}

__global__
void test(DataElem* data) {
	printf("On device, data[0]: value = %d\n", data[0].prop);
	data[0].prop += 10;
}

int main(void) {
	DataElem **data = NULL;
	cudaMallocManaged(&data, sizeof(DataElem*) * 2);

	data[0] = new DataElem[2];
	data[1] = new DataElem[3];

	data[0][0].prop = 1;
	data[0][1].prop = 2;
	data[1][0].prop = 3;
	data[1][1].prop = 4;
	data[1][2].prop = 5;

	add <<< 1, 1 >>> (data);
	CudaCheckError();

	printf("On host, data[0][0]: value = %d\n", data[0][0].prop);
	printf("On host, data[0][1]: value = %d\n", data[0][1].prop);
	printf("On host, data[1][0]: value = %d\n", data[1][0].prop);
	printf("On host, data[1][1]: value = %d\n", data[1][1].prop);
	printf("On host, data[1][2]: value = %d\n", data[1][2].prop);

	delete[] data[0];
	delete[] data[1];
	cudaFree(data);

	//	DataElem* data = new DataElem;
	//	data->prop = 1;
	//	test<<<1,1>>>(data);
	//	CudaCheckError();
	//	printf("On host, data[0]: value = %d\n", data[0].prop);
	//	delete data;

}
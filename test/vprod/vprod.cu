#include <iostream>

#include "../../power.hpp"

#define THREADS 512
#define BLOCKS	48

#define VSIZE	8192
#define N		BLOCKS * THREADS * VSIZE

__global__ void kernel(unsigned long long *x, unsigned long long *y, unsigned long long *z) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int vidx = idx * VSIZE;

	unsigned long long t = 0;

	for (int j = 0; j < 4; j++) {
	for (int i = 0; i < VSIZE; i++) {
			t += x[vidx + i] * y[vidx + i];
	}
	}
	

	z[idx] = t;
	

}

using namespace std;

int main(void) {

	unsigned long long *dev_x, *dev_y, *dev_z, *host;

	cudaEvent_t t1, t2;
	float time;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaHostAlloc((void**)&host, N * sizeof(unsigned long long), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_x, N * sizeof(unsigned long long));
	cudaMalloc((void**)&dev_y, N * sizeof(unsigned long long));
	cudaMalloc((void**)&dev_z, THREADS * BLOCKS * sizeof(unsigned long long));

	Power p(0,0);

	// Initialize
	memset(host, 0, N * sizeof(unsigned long long));

	// Start power measurement
	p.run();
	sleep(1);

	cudaMemcpy(dev_x, host, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, host, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, host, THREADS * BLOCKS * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	
	sleep(1);
	
	cudaEventRecord(t1, 0);
	
	kernel<<< dim3(BLOCKS), dim3(THREADS), 0, stream >>>(dev_x, dev_y, dev_z);

	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
	cudaEventElapsedTime(&time, t1, t2);

	sleep(1);

	cudaMemcpy(host, dev_y, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	
	sleep(1);
	p.halt();

	cudaEventDestroy(t1);
	cudaEventDestroy(t2);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	cudaFree(host);

	cudaStreamDestroy(stream);

	cin.get();

	cout << time << " ms" << endl;

	return 0;

}

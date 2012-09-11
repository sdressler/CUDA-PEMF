#include <iostream>

#include "../../power.hpp"

#define N 536870912
#define M 4

__global__ void kernel() {

}

using namespace std;

int main(void) {

	unsigned long long *dev, *host;

	cudaEvent_t t1, t2;
	float time1, time2;

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaHostAlloc((void**)&host, N * sizeof(unsigned long long), cudaHostAllocDefault);
	cudaMalloc((void**)&dev, N * sizeof(unsigned long long));

	Power p(0,0);

	// Make sure device is not in sleep mode
	cudaMemcpy(dev, host, 1024 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	
	// Start measurement
	p.run();
	sleep(1);

	cudaEventRecord(t1, 0);
	for (int i = 0; i < M; i++) {
		cudaMemcpy(dev, host, N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	}
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

	cudaEventElapsedTime(&time1, t1, t2);

	sleep(1);

	cudaEventRecord(t1, 0);
	for (int i = 0; i < M; i++) {
		cudaMemcpy(host, dev, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

	cudaEventElapsedTime(&time2, t1, t2);

	sleep(1);
	p.halt();

	cudaEventDestroy(t1);
	cudaEventDestroy(t2);

	cudaFree(dev);
	cudaFree(host);

	cin.get();

	//cout << time1 << " ms\t" << time2 << " ms" << endl;

	return 0;

}

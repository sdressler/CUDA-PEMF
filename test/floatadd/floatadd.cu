#include <iostream>
#include <vector>
#include <string>

#include "../../power.hpp"

#define THREADS	1024
#define BLOCKS	64
#define N THREADS * BLOCKS
//#define N 268435456
//#define M 5000000
//#define M 1048576
#define M 500000

__global__ void kernel(float *x, float *y) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float r1, r2, r3;

	r1 = x[idx];
	r2 = x[idx+1];

	for (int i = 0; i < M; i++) {
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
			r3 = r1 + r2;
			r2 = r3 + r1;
			r1 = r2 + r3;
	}

	y[idx] = r2;
	

}

using namespace std;

int main(void) {

	cudaSetDevice(0);

	float *dev_x, *dev_y, *host;

	cudaEvent_t t1, t2;
	float time;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaHostAlloc((void**)&host, N * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_x, N * sizeof(float));
	cudaMalloc((void**)&dev_y, N * sizeof(float));


	// Initialize
	memset(host, 0, N * sizeof(float));


	cudaMemcpy(dev_x, host, N * sizeof(float), cudaMemcpyHostToDevice);
	
//	for (int blocks = 1; blocks < BLOCKS + 1; blocks++) {
//		int iblocks = 2;
//		for (int threads = 1; threads < THREADS + 1; threads++) {

	cout << "# ThreadNo BlockNo Block1P Block1U BlockNo Block2P Block2U ... BlockNo BlockNP BlockNU" << endl;

//	for (int threads = 2; threads < THREADS + 1; threads = 2) {
		
//	int threads = 304;
//	int blocks = 17;

		// Start power measurement
		//p.run();
	for (int threads = 480; threads < 577; threads += 32) {
		cout << threads << " " << flush;
		for (int blocks = 64; blocks < 65; blocks++) {
//			Power p(0,2,false);
//			sleep(1);
//			cout << threads*blocks << " " << flush;
			cout << blocks << " " << flush;

			cudaEventRecord(t1, 0);
			kernel<<< dim3(blocks), dim3(threads) >>>(dev_x, dev_y);

//			sleep(1);
//			p.run();
			cudaEventRecord(t2, 0);
			
			cudaEventSynchronize(t2);
//			p.halt();
			cudaEventElapsedTime(&time, t1, t2);
//			p.getMean();

//			sleep(5);

//			p.halt();
//			cout << p.getPowerMean() << " " << p.getUtilizationMean() << " " << time - 1000.0 << " " << flush;
			cout << time << flush;
//			p.writeToFile("raw_power.dat", "raw_utilization.dat", blocks, threads);
		}
		cout << endl;
	}

	cudaMemcpy(host, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);
/*	
	vector<float> power = p.getPower();
	for (vector<float>::iterator i = power.begin(); i != power.end(); ++i) {
		cout << *i << " ";
	}
	cout << endl;

	vector<int> u = p.getUtilization();
	for (vector<int>::iterator i = u.begin(); i != u.end(); ++i) {
		cout << *i << " ";
	}
	cout << endl;
*/
	cudaEventDestroy(t1);
	cudaEventDestroy(t2);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(host);

	cudaStreamDestroy(stream);

	return 0;

}

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "../../power.hpp"

#define THREADS	1024
#define BLOCKS	64
#define N 268435456
#define M 500000
//#define M 1048576

__global__ void kernel(int *x, int *y) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r1, r2, r3;

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

	cudaSetDevice(2);

	int *dev_x, *dev_y, *host;

	cudaEvent_t t1, t2;
	float time;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	/* ---- long term power measurement ---- */
/*	Power pl(0, 0, true, 
		"/work/bzcdress/cuda/intadd_raw_power_long.dat", 
		"/work/bzcdress/cuda/intadd_raw_utilization_long.dat");
	pl.run();
*/
	cudaHostAlloc((void**)&host, N * sizeof(int), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_x, N * sizeof(int));
	cudaMalloc((void**)&dev_y, N * sizeof(int));


	// Initialize
	memset(host, 0, N * sizeof(int));

	cudaMemcpy(dev_x, host, N * sizeof(int), cudaMemcpyHostToDevice);
	
	cout << "# ThreadNo BlockNo Block1P Block1U BlockNo Block2P Block2U ... BlockNo BlockNP BlockNU" << endl;

	cout << setprecision(7);

	for (int threads = 511; threads < 515; threads++) {
		cout << threads << " ";
		for (int blocks = 64; blocks < BLOCKS + 1; blocks++) {
			Power pk(0,2,false);
			cout << blocks << " " << flush;

			cudaEventRecord(t1, 0);
			kernel<<< dim3(blocks), dim3(threads) >>>(dev_x, dev_y);

			sleep(1);
			pk.run();
			cudaEventRecord(t2, 0);
			cudaEventSynchronize(t2);

			pk.halt();
			cudaEventElapsedTime(&time, t1, t2);
			
			cout << pk.getPowerMean() << " " << pk.getUtilizationMean() << " " << time - 1000.0 << " " << flush;
/*			pk.writeToFile(
				"/work/bzcdress/cuda/intadd_raw_power.dat",
				"/work/bzcdress/cuda/intadd_raw_utilization.dat",
				blocks, threads);*/
		}
		cout << endl;
	}

	cudaMemcpy(host, dev_y, N * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventDestroy(t1);
	cudaEventDestroy(t2);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(host);

	cudaStreamDestroy(stream);

//	pl.halt();

	return 0;

}

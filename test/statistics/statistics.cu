#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "../../power.hpp"

#define VECDIM	16384
#define BLOCKS	64
#define THREADS	1024
#define ITER	1200

using namespace std;

__global__ void dotprod(float *vectors, float *dotprods) {

	int idx = blockDim.x * gridDim.x + threadIdx.x;

	for (int n = 0; n < ITER; n++) {
		
		float a = 0.0;
		
		for (int i = 0; i < VECDIM; i++) {
			a += pow(vectors[idx + i * VECDIM], 2.0f);
		}

		dotprods[idx] = sqrt(a);
	}

}

int main(void) {

	/* ---- Change device ---- */

	cudaSetDevice(2);

	/* ---- Initialize everything ---- */

	cudaEvent_t t1, t2;
	float time;

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	/* ---- long term power measurement ---- */
/*	Power pl(0, 1, true, 
		"/work/bzcdress/cuda/vecmet_raw_power_long.dat", 
		"/work/bzcdress/cuda/vecmet_raw_utilization_long.dat");
	pl.run();
*/
	float *dev_vectors, *host_vectors;
	float *dev_dotprods, *host_dotprods;

	/* ---- Alloc vectors ---- */
	host_dotprods = new float[THREADS * BLOCKS];
	host_vectors = new float[THREADS * BLOCKS * VECDIM];
	for (int i = 0; i < THREADS * BLOCKS * VECDIM; i++) {
		host_vectors[i] = 1.0;
	}

	cudaMalloc((void**) &dev_dotprods, BLOCKS * THREADS * sizeof(float));
	cudaMalloc((void**) &dev_vectors, BLOCKS * THREADS * VECDIM * sizeof(float));

	/* ---- Copy vectors to device */

	cudaMemcpy(dev_vectors, host_vectors, BLOCKS * THREADS * VECDIM * sizeof(float), cudaMemcpyHostToDevice);

	/* ---- Kernel call ---- */
	
	cout << setprecision(7);

	for (int i = 0; i < 1000; i++) {

			cout << i << " ";

			int blocks = 32;
			int threads = 512;

//	for (int threads = 511; threads < 515; threads++) {
//		cout << threads << " ";
//		for (int blocks = 64; blocks < BLOCKS + 1; blocks++) {
			Power pk(0,2,false);
//			cout << blocks << " ";

			cudaEventRecord(t1, 0);
			dotprod<<< blocks, threads >>>(dev_vectors, dev_dotprods);

			sleep(1);
			pk.run();
			cudaEventRecord(t2, 0);
			cudaEventSynchronize(t2);
			
			pk.halt();
			cudaEventElapsedTime(&time, t1, t2);

//			cout << pk.getPowerMean() << " " << pk.getUtilizationMean() << " " << time - 1000.0 << " " << flush;
			pk.writeToFile(
				"/work/bzcdress/cuda/statistics_raw_power.dat",
				"/work/bzcdress/cuda/statistics_raw_utilization.dat",
				32, 512);
//		}
//		cout << endl;
//	}
	}
	cout << endl;

	/* ---- Backcopy ---- */

	cudaMemcpy(host_dotprods, dev_dotprods, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_vectors);
	cudaFree(dev_dotprods);

	delete[] host_dotprods;
	delete[] host_vectors;

//	pl.halt();

	return 0;

}

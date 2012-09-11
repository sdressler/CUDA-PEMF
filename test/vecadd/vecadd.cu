#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "../../power.hpp"

#define BLOCKS	64
#define BSTART  1
#define THREADS	1024
#define TSTART	1
#define VECDIM BLOCKS * THREADS * 4096

using namespace std;

__global__ void vecadd(double ma, double mb, double *vec_a, double *vec_b) {

	int idx = blockIdx.x * gridDim.x + threadIdx.x;
	int L = blockDim.x * gridDim.x;

	for (int i = 0 ; i < VECDIM / L; i++) {
		vec_a[idx + i] += pow(vec_a[idx + L], vec_b[idx + L]);
	}

}

int main(void) {

	/* ---- Change device ---- */

	cudaSetDevice(0);

	/* ---- Initialize everything ---- */

	cudaEvent_t t1, t2;
	float time;

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	/* ---- long term power measurement ---- */
//	Power pl(0, 1, true, 
//		"/work/bzcdress/cuda/vecmet_raw_power_long.dat", 
//		"/work/bzcdress/cuda/vecmet_raw_utilization_long.dat");
//	pl.run();

	double *dev_vec_a, *dev_vec_b;
	double *hst_vec_a, *hst_vec_b;

	/* ---- Alloc vectors ---- */
	hst_vec_a = new double[VECDIM];
	hst_vec_b = new double[VECDIM];

	for (int i = 0; i < VECDIM; i++) {
		hst_vec_a[i] = 1.0;
		hst_vec_b[i] = 1.0;
	}
	
	cudaMalloc((void**) &dev_vec_a, VECDIM * sizeof(double));
	cudaMalloc((void**) &dev_vec_b, VECDIM * sizeof(double));

	/* ---- Copy vectors to device */

	cudaMemcpy(dev_vec_a, hst_vec_a, VECDIM * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec_b, hst_vec_b, VECDIM * sizeof(double), cudaMemcpyHostToDevice);

	/* ---- Kernel call ---- */
	
	cout << setprecision(7);

	for (int threads = TSTART; threads < THREADS + 1; threads++) {
		cout << threads << " ";
		for (int blocks = BSTART; blocks < BLOCKS + 1; blocks++) {
			Power pk(0,0,false);
			cout << blocks << " ";

			cudaEventRecord(t1, 0);
			vecadd<<< blocks, threads >>>(2.0, 2.0, dev_vec_a, dev_vec_b);

			//sleep(1);i
			pk.run();
			cudaEventRecord(t2, 0);
			cudaEventSynchronize(t2);
			
			pk.halt();
			cudaEventElapsedTime(&time, t1, t2);

			cout << pk.getPowerMean() << " " << pk.getUtilizationMean() << " " << time << " " << flush;
			pk.writeToFile(
				"/work/bzcdress/cuda/vecadd_raw_power_from_1.dat",
				"/work/bzcdress/cuda/vecadd_raw_utilization_from_1.dat",
				blocks, threads);
		}
		cout << endl;
	}

	/* ---- Backcopy ---- */

//	cudaMemcpy(host_dotprods, dev_dotprods, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_vec_a, dev_vec_a, VECDIM * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_vec_a);
	cudaFree(dev_vec_b);

	delete[] hst_vec_a;
	delete[] hst_vec_b;

//	pl.halt();

	return 0;

}

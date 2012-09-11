#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "../../power.hpp"

#define BLOCKS	64
#define BSTART  64
#define THREADS	448
#define TSTART	256
#define TSTOP	448
#define TINC	32
#define VECDIM  8192
#define ITER    128
#define DEV		0

using namespace std;

__global__ void vecnorm(double *vec, double *vec_norm) {

	int idx = blockIdx.x * gridDim.x + threadIdx.x;

	double norm = 0.0;
	for (int n = 0; n < ITER; n++) {
		for (int i = 0 ; i < VECDIM; i++) {
			norm += pow(vec[idx + i], 2.0); 
		}
	}

	vec_norm[idx] = sqrt(norm);

}

int main(void) {

	/* ---- Change device ---- */

	cudaSetDevice(DEV);

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

	double *dev_vec, *dev_vec_norm;
	double *hst_vec, *hst_vec_norm;

	/* ---- Alloc vectors ---- */
	hst_vec = new double[VECDIM * THREADS * BLOCKS];
	hst_vec_norm = new double[THREADS * BLOCKS];

	for (int i = 0; i < VECDIM * THREADS * BLOCKS; i++) {
		hst_vec[i] = 1.0;
	}
	
	cudaMalloc((void**) &dev_vec, VECDIM * THREADS * BLOCKS * sizeof(double));
	cudaMalloc((void**) &dev_vec_norm, THREADS * BLOCKS * sizeof(double));

	/* ---- Copy vectors to device */

	cudaMemcpy(dev_vec, hst_vec, VECDIM * THREADS * BLOCKS * sizeof(double), cudaMemcpyHostToDevice);

	/* ---- Kernel call ---- */
	
	cout << setprecision(7);

	for (int threads = TSTART; threads < TSTOP + TINC; threads += TINC) {
		cout << threads << " ";
		for (int blocks = BSTART; blocks < BLOCKS + 1; blocks++) {
			Power pk(0,DEV,false);
			cout << blocks << " ";

			cudaEventRecord(t1, 0);
			vecnorm<<< blocks, threads >>>(dev_vec, dev_vec_norm);

			//sleep(1);i
			pk.run();
			cudaEventRecord(t2, 0);
			cudaEventSynchronize(t2);
			
			pk.halt();
			cudaEventElapsedTime(&time, t1, t2);

			cout << time << flush;
//			cout << pk.getPowerMean() << "\t" << pk.getUtilizationMean() << "\t\t" << time << " " << flush;
//			pk.writeTioFile(
//				"/work/bzcdress/cuda/vecnorm_raw_power_from_1.dat",
//				"/work/bzcdress/cuda/vecnorm_raw_utilization_from_1.dat",
//				blocks, threads);
		}
		cout << endl;
	}

	/* ---- Backcopy ---- */

//	cudaMemcpy(host_dotprods, dev_dotprods, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_vec, dev_vec, THREADS * BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_vec);
	cudaFree(dev_vec_norm);

	delete[] hst_vec;
	delete[] hst_vec_norm;

//	pl.halt();

	return 0;

}

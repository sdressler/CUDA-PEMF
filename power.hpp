#ifndef POWER_HPP
#define POWER_HPP

#include "threads.hpp"

#include <time.h>
#include <nvml.h>
#include <stdio.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

class Power : public Thread {

private:
	nvmlDevice_t dev;
	unsigned int x;
	vector<float> power;
	vector<int> utilization;
	
	bool directWriteToFile;

	string powerFile;
	string utilizationFile;

public:
	Power(
		unsigned int _id,
		unsigned int _did,
		bool _directWriteToFile,
		string _powerFile = "",
		string _utilizationFile = "") : Thread(_id) {
	
			x = 0;
	
			directWriteToFile = _directWriteToFile;
			powerFile = _powerFile;
			utilizationFile = _utilizationFile;
	
			nvmlInit();
			nvmlDeviceGetHandleByIndex(_did, &dev);
	}

	void worker() {

		unsigned int p;
		nvmlUtilization_t u;
		ofstream fp, fu;

		if (directWriteToFile) {
			fp.open(powerFile.c_str(), ios::out | ios::trunc);
			fp << setprecision(7);
			fu.open(utilizationFile.c_str(), ios::out | ios::trunc);
			fu << setprecision(7);
		}

		while (!mtx_stopreq) {
			usleep(20000);
		
			nvmlDeviceGetPowerUsage(dev, &p);
			nvmlDeviceGetUtilizationRates(dev, &u);
		
			if (!directWriteToFile) {
				power.push_back((float)p/1000.0);
				utilization.push_back(u.gpu);
			} else {
				fp << (float)p/1000.0 << " ";
				fu << u.gpu << " ";
			}
			
		}

		if (directWriteToFile) {

			fp << flush;
			fp.close();

			fu << flush;
			fu.close();
		}
	}

	double getPowerMean() {

		double s = 0.0;
		for (vector<float>::iterator it = power.begin(); it != power.end(); ++it) {
			s += (double)*it;
		}
		return s / (double)power.size();

//		gsl_vector_float_const_view gsl_vpwr = gsl_vector_float_const_view_array(&power[0], power.size()-100);
//		return gsl_stats_float_mean(gsl_vpwr.vector.data, 1, power.size());
	}

	double getUtilizationMean() {

		double s = 0.0;
		for (vector<int>::iterator it = utilization.begin(); it != utilization.end(); ++it) {
			s += (double)*it;
		}
		return s / (double)utilization.size();

//		gsl_vector_int_const_view gsl_vutil = gsl_vector_int_const_view_array(&utilization[0], utilization.size()-100);
//		return gsl_stats_int_mean(gsl_vutil.vector.data, 1, utilization.size());
	}

	vector<float> getPower() { return power; }
	vector<int> getUtilization() { return utilization; }

	void writeToFile(string s1, string s2, int block, int thread) {
		ofstream f(s1.c_str(), ios::out | ios::app);

		f << thread << " " << block << " ";
		f << setprecision(7);
		for (vector<float>::iterator it = power.begin(); it != power.end(); ++it) {
			f << *it << " ";
		}
		f << endl;
		f << flush;

		f.close();

		f.open(s2.c_str(), ios::out | ios::app);

		f << thread << " " << block << " ";
		f << setprecision(7);
		for (vector<int>::iterator it = utilization.begin(); it != utilization.end(); ++it) {
			f << *it << " ";
		}
		f << endl;
		f << flush;

		f.close();
	}

/*
	void getMean() {
		
//		cout << power.size() << " " << utilization.size() << endl;

		gsl_vector_float_const_view gsl_vpwr = gsl_vector_float_const_view_array(&power[0], power.size()-100);
		gsl_vector_int_const_view gsl_vutil = gsl_vector_int_const_view_array(&utilization[0], utilization.size()-100);

		cout
			<< gsl_stats_float_mean(gsl_vpwr.vector.data, 1, power.size()) << " "
			<< gsl_stats_int_mean(gsl_vutil.vector.data, 1, utilization.size()) << endl;
	}
*/
};

#endif /* POWER_HPP */

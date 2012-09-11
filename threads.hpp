#include <iostream>
#include <stdexcept>

extern "C" {
	#include <pthread.h>
}

using namespace std;

class Thread {
protected:
	volatile bool mtx_stopreq;
	volatile bool mtx_run;

private:
	pthread_t _thrd;

	unsigned int tid;

	static void* do_work(void *arg) {
		Thread *t = static_cast<Thread*>(arg);
		t->worker();
	}

	virtual void worker() = 0;

public:
	Thread(unsigned int _tid) {
		tid = _tid;
		mtx_stopreq = false;
		mtx_run = false;
	};

	void run() {
//		cout << "Run" << endl;
		//assert(mtx_run == false);
		if (mtx_run) {
			throw std::runtime_error("Error: Start requested but thread already running.");
		}
		mtx_run = true;
		if (pthread_create(&_thrd, NULL, do_work, (void*)this) != 0) {
			throw std::runtime_error("Error in pthread creation.");
		}
	}

	void halt() {
//		cout << "Halt" << endl;
		//assert(mtx_run == true);
		if (!mtx_run) {
			throw std::runtime_error("Error: Stop requested but thread not running.");
		}
		mtx_run = false;
		mtx_stopreq = true;
		pthread_join(_thrd, 0);
	}

};

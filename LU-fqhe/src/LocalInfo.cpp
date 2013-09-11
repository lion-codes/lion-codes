/*
 *  Copyright 2013 William J. Brouwer, Pierre-Yves Taunay
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include <omp.h>
#include <mpi.h>
#include <exception>
#include <iostream>

#include "LocalInfo.h"

using namespace MPI;
using namespace std;

LocalInfo::LocalInfo() {
	// Get the rank of the node
	try {
		rank = COMM_WORLD.Get_rank();
#ifdef __VERBOSE__
		cout << "INFO \t My rank is " << rank << endl;
#endif 
	}
	catch (MPI::Exception& e) {
		cerr << "ERROR \t " << e.Get_error_code() << " - " << e.Get_error_string()	<< endl;
		throw exception();
	}

	// Get the number of CPU on that node
	#pragma omp parallel
		cpuCnt = omp_get_num_threads();

#ifdef __VERBOSE__
	cout << "INFO \t Node " << rank << "\t Found " << cpuCnt << " CPU(s) available" << endl;
#endif

	// Are there any GPU connected on that node ?
	try {
		gpuInfo = new GpuInfo();
#ifdef __VERBOSE__
		cout << "INFO \t Node " << rank << "\t Found " << gpuInfo->getGpuCnt() << " GPU(s) available" << endl;
#endif	
	}
	catch(std::bad_alloc& e) {
		cerr << "ERROR \t Failed to allocate memory " << e.what() << endl;
		throw exception();
	}
}

int LocalInfo::getRank() {
	return rank;
}

int LocalInfo::getCpuCnt() {
	return cpuCnt;
}

GpuInfo* LocalInfo::getGpuInfo() {
	return gpuInfo;
}

LocalInfo::~LocalInfo() {
	if( gpuInfo != NULL ) {
	 	delete gpuInfo;
		gpuInfo = NULL;
	}	
}


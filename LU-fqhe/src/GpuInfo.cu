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


#include <iostream>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cuda.h>
#include <mpi.h>
#include <cmath>

#include "GpuInfo.h"

using namespace std;


GpuInfo::GpuInfo() {
	cudaGetDeviceCount(&gpuCnt);

	int cpuCnt;

	#pragma omp parallel
		cpuCnt = omp_get_num_threads();

	// Takes care of the case where we have more GPUs than CPUs
	gpuCnt = min(gpuCnt,cpuCnt);
	#ifdef __NOGPU__
	gpuCnt = 0;
	#endif

	int rank = MPI::COMM_WORLD.Get_rank();

	int th_num;

	if(this->gpuCnt > 0) {
		cudaDeviceProp localProp;
		#pragma omp parallel private(th_num,localProp)
		{
			th_num = omp_get_thread_num();

			if(th_num < gpuCnt) {
					cudaGetDeviceProperties(&localProp,th_num);	// Local property for each device on a given node
					#pragma omp critical
					{
						gpuProp.push_back(SingleGpuInfo(th_num,localProp.warpSize,localProp.maxThreadsPerBlock,
								localProp.maxThreadsDim[0],localProp.maxGridSize[0],localProp.totalGlobalMem)); // Add 1 GPU at a time to avoid race conditions
#ifdef __VERBOSE__
						cout << "INFO \t Node " << rank << "\t GPU " << th_num << " Characteristics" << endl;
						cout << "INFO \t Node " << rank << "\t Warp size \t\t\t" 			<< localProp.warpSize << endl;
						cout << "INFO \t Node " << rank << "\t Max. threads per block \t" << localProp.maxThreadsPerBlock << endl;
						cout << "INFO \t Node " << rank << "\t Max. threads in 1 dim \t\t" 	<< localProp.maxThreadsDim[0] << endl;
						cout << "INFO \t Node " << rank << "\t Max. blocks in 1 dim \t\t" 	<< localProp.maxGridSize[0] << endl;
						cout << "INFO \t Node " << rank << "\t Total global mem (MB) \t\t" << (float)localProp.totalGlobalMem/(1024.f*1024.f) << endl;
#endif
					}
			}
		}
	}
}


int GpuInfo::getGpuCnt() {
	return gpuCnt;
}


SingleGpuInfo GpuInfo::getSingleGpuInfo(int N) {

	try {
		return gpuProp.at(N);
	}
	catch (const std::out_of_range& oor) {
		std::cerr << "ERROR: Out of Range error: " << oor.what() << '\n';
	}
}

GpuInfo::~GpuInfo() {
	gpuProp.clear();	
	vector<SingleGpuInfo>().swap(gpuProp);
}

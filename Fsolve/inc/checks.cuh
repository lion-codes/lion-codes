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



#ifndef __CHECKS_CUH_
#define __CHECKS_CUH_

#include <cuda.h>



void inline cudaCheckError(const char* msg) {
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err  )  {
			cerr << "ERROR \t CUDA failed at [ " << msg << " ] with [ " << cudaGetErrorString(err) << " ] " << endl;
			throw std::invalid_argument( msg );
	}		

}	
#endif


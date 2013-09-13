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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <cuda.h>

#include "cuint.h"

namespace cuint {

	void STRAPZ(float *V,float *out,float dx, int N) {
		
		//// Cast thrust pointer
		thrust::device_ptr<float> th_ptr = thrust::device_pointer_cast(V);
		thrust::device_vector<float> th_V (th_ptr,th_ptr + N);

		//// Perform reduction
		*out = thrust::reduce(	th_V.begin() + 1,
					th_V.end() -1,
					0.5f*(th_V[0] + th_V[N-1]),
					thrust::plus<float>() );
		
		//// Multiply by h
		*out *= dx;
	}	

	void DTRAPZ(double *V,double *out,double dx, int N) {
		
		//// Cast thrust pointer
		thrust::device_ptr<double> th_ptr = thrust::device_pointer_cast(V);
		thrust::device_vector<double> th_V (th_ptr,th_ptr + N);

		//// Perform reduction
		*out = thrust::reduce(	th_V.begin() + 1,
					th_V.end() -1,
					0.5f*(th_V[0] + th_V[N-1]),
					thrust::plus<double>() );
		
		//// Multiply by h
		*out *= dx;
	}	

}


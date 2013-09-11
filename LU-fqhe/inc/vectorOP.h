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

#ifndef VECTOROP_H_
#define VECTOROP_H_

#include <cuda.h>


namespace vOP {

	__inline__ __device__ bool geq(float A, int indexA, float B, int indexB);
	__inline__ __device__ void findVectorMaxima(volatile float2 * input, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU);
	__inline__ __device__ void findVectorMaximaKey( volatile float2 * input, volatile int * indices, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU);
	__inline__ __device__ void findVectorProduct(volatile float2 * input, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU);

}	
#endif /* VECTOROP_H_ */

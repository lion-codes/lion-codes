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

#ifndef MKMATRIXELEMENTS_H_
#define MKMATRIXELEMENTS_H_

#include <cuda.h>

#if 0

namespace mkElem {


#ifdef __DPREC__

#else


	__device__ float2 calculateElement(int rowIndex, int colIndex, int * perm);
	__global__ void makeMatrices(int offset, float2 * devMatrices, short * devPermutations);

	void makeMatrixElements(int rank,char * hostname,
								float * phi,float * theta,
									float k1,float k2,float r1,float r2,int mu,	int nu,	int Q,
										int myPerms,	short * devPermutations,void * devMatrices);

	void setupM_Matrix ( float * theta, float * phi, float2 * u, float2 * v, int Q, float2 * buf );

#endif


}
#endif


#endif /* MKMATRIXELEMENTS_H_ */

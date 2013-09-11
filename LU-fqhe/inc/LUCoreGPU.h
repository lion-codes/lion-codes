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

#ifndef LUCOREGPU_H_
#define LUCOREGPU_H_

#include <cuda.h>

#include "RunInfo.h"

#define perm_index		(vectorIndex + myMatrix * matSide_GPU + blockIdx.x * NUM_MATRICES * matSide_GPU * loadSize_GPU)
#define diag_index 		(memoryStride + vectorIndex * loadSize_GPU + myMatrix * matSide_GPU * loadSize_GPU + vectorIndex)
#define col_j_index 	(memoryStride + vectorIndex * loadSize_GPU + myMatrix * matSide_GPU * loadSize_GPU + j)
#define col_k_index 	(memoryStride + k + myMatrix * matSide_GPU * loadSize_GPU + vectorIndex * loadSize_GPU)
#define col_i_index 	(memoryStride + i + myMatrix * matSide_GPU * loadSize_GPU + vectorIndex * loadSize_GPU)
#define row_j_index 	(memoryStride + vectorIndex + myMatrix * matSide_GPU * loadSize_GPU + j * loadSize_GPU)
#define row_i_index 	(memoryStride + vectorIndex + myMatrix * matSide_GPU * loadSize_GPU + i * loadSize_GPU)
#define row_k_index 	(memoryStride + vectorIndex + myMatrix * matSide_GPU * loadSize_GPU + k * loadSize_GPU)
#define diag_j_index	(memoryStride + j + myMatrix * matSide_GPU * loadSize_GPU + j * loadSize_GPU)
#define mat_00_index	(memoryStride + vectorIndex + myMatrix * matSide_GPU * loadSize_GPU)
#define vec_1_index		(vectorIndex + (myMatrix-1) * scrSpace_GPU)
#define vec_index 		(vectorIndex + myMatrix * scrSpace_GPU)
#define scale_index 	(vectorIndex + myMatrix * matSide_GPU)
#define vec_k_scr		(k + myMatrix * scrSpace_GPU)
#define vec_i_sca		(i + myMatrix * matSide_GPU)
#define vec_j_sca		(j + myMatrix * matSide_GPU)
#define vec_00_scr		(myMatrix * scrSpace_GPU)

namespace LUCoreGPU {

#ifdef __DPREC__

#else
		void CALL(float2 **d_Matrices, int nMat, RunInfo *rInfo, int rank, int th_num, cudaStream_t *stream);
		__global__ void LUDecomposition (int vestige, float2 *inputMatrices);

		void globalReduction(float2 **d_Matrices, float2 **d_Reduction, int nMat,int rank,int th_num,cudaStream_t stream);
		__global__ void reductionGlobalSum(int vestige, float2 * input,float2 *output, int it);

#endif

}

#endif /* LUCOREGPU_H_ */

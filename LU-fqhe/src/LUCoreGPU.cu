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

#include <cuda.h>

#include "LUCoreGPU.h"
#include "RunInfo.h"
#include "vectorOP.h"
#include "vectorOP.cu"

#include "checks.cuh"

#define RED_SUM_SIZE 1024

using namespace std;

namespace LUCoreGPU {
	__device__ __constant__ int nMperBl_GPU, matSide_GPU, scrSpace_GPU, scrSpace2_GPU, loadSize_GPU, warpPerMat_GPU;

#ifdef __DPREC__
	void CALL(double2 *d_localList, int nMat, RunInfo *rInfo) {
#else
		void CALL(float2 **d_Matrices, int nMat, RunInfo *rInfo, int rank, int th_num, cudaStream_t *stream) {
#endif
			// Power of 2 above and below matSize
			int scrSpace = rInfo->getScrSpace();
			int scrSpace2= rInfo->getScrSpace2();

			// Number of matrices per block, max number of blocks, load size
			int nMperBl 	= rInfo->getNMperBl();
			int maxBlocks 	= rInfo->getMaxBlocks();
			int loadSize	= rInfo->getLoadSize();
			int warpPerMat	= rInfo->getWarpPerMat();

			// Matrix side
			int matSide		= rInfo->getMatSide();

			cudaMemcpyToSymbol(nMperBl_GPU	,&nMperBl,sizeof(nMperBl));
#ifdef __DEBUG__	
			cudaCheckError("nMperBl memcpy");
#endif	
			cudaMemcpyToSymbol(matSide_GPU	,&matSide,sizeof(matSide));
#ifdef __DEBUG__			
			cudaCheckError("matSide memcpy");
#endif
			cudaMemcpyToSymbol(scrSpace_GPU	,&scrSpace,sizeof(scrSpace));
#ifdef __DEBUG__			
			cudaCheckError("scrSpace memcpy");
#endif
			cudaMemcpyToSymbol(scrSpace2_GPU,&scrSpace2,sizeof(scrSpace2));
#ifdef __DEBUG__			
			cudaCheckError("scrSpace2 memcpy");
#endif
			cudaMemcpyToSymbol(loadSize_GPU	,&loadSize,sizeof(loadSize));
#ifdef __DEBUG__			
			cudaCheckError("loadSize memcpy");
#endif
			cudaMemcpyToSymbol(warpPerMat_GPU,&warpPerMat,sizeof(warpPerMat));
#ifdef __DEBUG__			
			cudaCheckError("warpPerMat memcpy");
#endif

			dim3 threads, blocks;
			threads.x = nMperBl * scrSpace;

			int * devPerms = NULL;


			// TODO : Replace if statements with try / catch
			if ((nMat / nMperBl) < maxBlocks-1){

				//TODO incorporate more perms
				blocks.x=(nMat / nMperBl)+1;

				int vestige = nMat - ((blocks.x-1) * nMperBl);
#ifdef __VERBOSE__
				cout << "INFO \t Node " << rank << " Thread " << th_num << "\t Queuing LU Decomposition kernel with " << blocks.x << " blocks, " << threads.x << " threads" << endl;
#endif
				// kernel
				LUDecomposition <<< blocks, threads,nMperBl*(1+matSide+5*scrSpace)*sizeof(float),*stream >>> (vestige,d_Matrices[th_num]);
				//	LUDecomposition <<< blocks, threads,0,*stream >>> (vestige,d_Matrices[th_num]);
				//	LUDecomposition <<< blocks, threads >>> (vestige,d_Matrices[th_num]);
#ifdef __DEBUG__			
				cudaCheckError("LU Decomposition kernel");
#endif

			} else {
				//		fprintf(stderr,"%.24s EXITING; exceeded avail blocks on GPU on host %s with rank %i\n",hostname,rank);
				exit(1);
			}
		}



#ifdef __DPREC__

#else
		void globalReduction(float2 **d_Matrices, float2 **d_Reduction, int nMat,int rank,int th_num, cudaStream_t stream){
#endif
			// Compute the sum for fqhe wf
			// we've performed the multiplication already

			nMat /= 2;

			time_t t = time(0);
			dim3 threads,blocks;

			// TODO: Change number of threads to flexible value
			threads.x 	= RED_SUM_SIZE;

			int vestige=0, k=1, i=0;

			float2 *d_Output = d_Reduction[th_num];
			float2 *d_Input = d_Matrices[th_num];
			float2 *tmp;


			while (k != -1){


				blocks.x 	= ( nMat > threads.x ) ? nMat / threads.x +1 : 1;
				vestige 	= nMat - (blocks.x-1) * threads.x;

#ifdef __VERBOSE__
#pragma omp critical
				cout << "INFO \t Node " << rank << " Thread " << th_num << "\t Queuing the global reduction kernel with " << blocks.x << " blocks, " << threads.x << endl;
#endif
				if ((i%2)==1 )
				{
					tmp = d_Output;
					d_Output = d_Input;
					d_Input = tmp;
				}	

				reductionGlobalSum<<<blocks,threads,0,stream>>>(vestige,d_Input,d_Output,i);
				i++;

				nMat = blocks.x;
				if (blocks.x==1) k=-1;
			}
		}

#ifdef __DPREC__
		__global__ void LUDecomposition (int vestige, double2 * inputMatrices){
#else
			__global__ void LUDecomposition (int vestige, float2 * inputMatrices){
#endif
				// Broadcast the constant memory to registers
				int lmem_nMperBl = nMperBl_GPU;
				int lmem_matSide = matSide_GPU;
				int lmem_scrSpace = scrSpace_GPU;
				int lmem_scrSpace2 = scrSpace2_GPU;
				int lmem_loadSize = loadSize_GPU;
				int	lmem_warpPerMat = warpPerMat_GPU;

#ifdef __DPREC__
				double2 sum, dum,tmp;
				__shared__ double				sign [ lmem_nMperBl ]; // permutation sign, for determinant
				volatile __shared__  double   	scale [ lmem_nMperBl * lmem_matSide ]; // scaling information, store largest value for each row
				volatile __shared__ double2		reduce [ lmem_nMperBl * lmem_scrSpace ]; // a reduction buffer
				volatile __shared__ double2		vectors [ lmem_nMperBl * lmem_scrSpace ]; // scratch space for main steps in algorithm
#else
				float2 sum, dum,tmp;
				float den;
				extern __shared__ float BANK[];
				volatile float *sign 	= (float*)BANK;
				volatile float *scale 	= (float*)&sign[lmem_nMperBl];
				volatile float2 *reduce = (float2*)&scale[lmem_nMperBl*lmem_matSide];
				volatile float2 *vectors= (float2*)&reduce[lmem_nMperBl*lmem_scrSpace];
				/*	
					__shared__ 		float 	sign[4];
					__shared__ volatile	float 	scale[4*16];
					__shared__ volatile	float2 	reduce[4*16];
					__shared__ volatile	float2 	vectors[4*16];
					__shared__ volatile	int	indices[4*16];*/
#endif

				volatile int *indices = (int*)&vectors[lmem_nMperBl*lmem_scrSpace];

				// Index to matrix for processing
				int myMatrix	= threadIdx.x / lmem_scrSpace;

				// Index to vector for processing
				int vectorIndex	= threadIdx.x % lmem_scrSpace;

				// Initialize permutation signs
#ifdef __DPREC__
				sign[threadIdx.x % lmem_nMperBl]=1.0;
#else
				sign[threadIdx.x % lmem_nMperBl]=1.0f;
#endif
				// book-keeping; if last block, only need vestige * lmem_scrSpace threads
				if (blockIdx.x == (gridDim.x-1))
					if (threadIdx.x >= vestige * lmem_scrSpace)
						return;

				// offset for this block
				int memoryStride 	= ( blockIdx.x * lmem_nMperBl * lmem_matSide * lmem_loadSize );

				// initialize memory
				vectors [ threadIdx.x ].x = -1e24;
				vectors [ threadIdx.x ].y = -1e24;


				// Determine scaling information
				for (int i=0; i < lmem_matSide; ++i){
					// load memory
					if ( vectorIndex < lmem_loadSize ){
						vectors [ vec_index ].x = inputMatrices [ row_i_index ].x;
						vectors [ vec_index ].y = inputMatrices [ row_i_index ].y;
					}
					__syncthreads();

					// find maxima
					vOP::findVectorMaxima ( vectors, vectorIndex, myMatrix, scrSpace_GPU, scrSpace2_GPU );

					// write scaling information
					if ( vectorIndex ==i ){
						scale [ scale_index ] = abs ( vectors [ vec_00_scr ].x);
					}
				}

				// initialize memory
				vectors [ threadIdx.x ].x = 0.0f;
				vectors [ threadIdx.x ].y = 0.0f;

				//__syncthreads();

				// main loops
				float2 tmpr;
				float2 tmpl;

				int myWarp = vectorIndex / 32;
				for (int j=0; j<lmem_matSide; j++){

					// load the j column to shared
					if ( vectorIndex < lmem_matSide ){
						vectors [ vec_index ].x 	= inputMatrices [ col_j_index ].x;
						vectors [ vec_index ].y 	= inputMatrices [ col_j_index ].y;
					}

					// update the j column

					__syncthreads();

					for (int i=0; i<lmem_warpPerMat; i++){

						if (myWarp==i){

							if ( vectorIndex < j ){
								sum.x = vectors[vec_index].x;
								sum.y = vectors[vec_index].y;

								for (int k=0; k< 512; k++){

									if (k >= vectorIndex) break; 

									tmpl.x = inputMatrices[col_k_index].x;
									tmpl.y = inputMatrices[col_k_index].y;

									tmpr.x = vectors [ vec_k_scr ].x;
									tmpr.y = vectors [ vec_k_scr ].y;

									sum.x -= (tmpl.x * tmpr.x - tmpl.y * tmpr.y);
									sum.y -= (tmpl.y * tmpr.x + tmpl.x * tmpr.y);

									vectors[vec_index].x = sum.x;
									vectors[vec_index].y = sum.y;
								}
							}
						}

						__syncthreads();
					}

					__syncthreads();
					for (int i=0; i<lmem_warpPerMat; i++){

						if(myWarp==i) {
							if ((vectorIndex >=j) && (vectorIndex < lmem_matSide)){
								sum.x = vectors[vec_index].x;
								sum.y = vectors[vec_index].y;

								for (int k=0; k< j; k++){

									tmpl.x = inputMatrices [ col_k_index ].x;
									tmpl.y = inputMatrices [ col_k_index ].y;

									tmpr.x = vectors [ vec_k_scr ].x;
									tmpr.y = vectors [ vec_k_scr ].y;

									sum.x -= (tmpl.x * tmpr.x - tmpl.y * tmpr.y);
									sum.y -= (tmpl.y * tmpr.x + tmpl.x * tmpr.y);
									vectors[vec_index].x=sum.x;
									vectors[vec_index].y=sum.y;
								}
							}
						}
					}
					__syncthreads();

					// write j column back to global
					if ( vectorIndex < lmem_matSide ){
						inputMatrices [ col_j_index ].x = vectors [ vec_index ].x;
						inputMatrices [ col_j_index ].y = vectors [ vec_index ].y;
					}

					// initialize shared memory
					reduce [ threadIdx.x ].x =  -1e24;
					reduce [ threadIdx.x ].y =  -1e24;
					//__syncthreads();

					if ((vectorIndex >= j) &&  (vectorIndex < lmem_matSide)){

						// init for pivot search
						reduce [ vec_index - j ].x 	= abs ( vectors [ vec_index ].x ) /  scale [ scale_index ];
						indices [ vec_index - j ] 	= vectorIndex;
					}

					__syncthreads();
					vOP::findVectorMaximaKey ( reduce, indices, vectorIndex, myMatrix, scrSpace_GPU, scrSpace2_GPU );
					//__syncthreads();

					// possible row swap
					if (j != indices [ vec_00_scr ]){


						if (vectorIndex < lmem_loadSize){
							int i = indices [ vec_00_scr ];

							// each thread swaps one row element with another row element

							sum 				= inputMatrices [ row_i_index ];
							inputMatrices [ row_i_index ] 	= inputMatrices [ row_j_index ];
							inputMatrices [ row_j_index ] 	= sum;

							if (vectorIndex==0){
								scale [ vec_i_sca ] 	= scale [ vec_j_sca ];
								sign [ myMatrix ] 	*= -1.0f;
							}
						}
					}

					__syncthreads();

					// final scaling
					if ( j != lmem_matSide-1){

						dum = inputMatrices [ diag_j_index ];

						if ((vectorIndex >= j+1) && (vectorIndex < lmem_matSide)){

							tmp 				= inputMatrices [ col_j_index ];

							// Perform division tmp/dum
							den				= dum.x*dum.x + dum.y*dum.y;
							tmpl.x				= (tmp.x*dum.x + tmp.y*dum.y)/den;
							tmpl.y				= (tmp.y*dum.x - tmp.x*dum.y)/den;
							tmp.x				= tmpl.x;
							tmp.y				= tmpl.y;

							inputMatrices [ col_j_index ] 	= tmp;
						}
					}
					__syncthreads();
				}// end j loops


#ifdef WRITE_DET
				// init
				vectors [ vec_index].x = 1.0f;
				vectors [ vec_index].y = 0.0f;

				// load diags
				if (vectorIndex < lmem_matSide ){
					vectors [ vec_index ].x = inputMatrices [ diag_index ].x;
					vectors [ vec_index ].y = inputMatrices [ diag_index ].y;
				}
				__syncthreads();

				// pop in sign
				if (vectorIndex == 0 ){
					vectors [ vec_00_scr ].x 	*= sign [ myMatrix ];
					vectors [ vec_00_scr ].y 	*= sign [ myMatrix ];
				}
				__syncthreads();

#ifdef PERFORM_PROD
				// every second row will contain product of two dets

				if ((myMatrix %2) ==1){
					// compiler complains about 'no copy constructor for float2'
					// so doing this explicitly
#ifdef __DPREC__
					double2 tmpA, tmpB;
#else
					float2 tmpA, tmpB;
#endif

					tmpA.x = vectors [ vec_1_index].x;
					tmpA.y = vectors [ vec_1_index].y;
					tmpB.x = vectors [ vec_index].x;
					tmpB.y = vectors [ vec_index].y;

					vectors [ vec_1_index ].x = tmpA.x*tmpB.x-tmpA.y*tmpB.y;
					vectors [ vec_1_index ].y = tmpA.y*tmpB.x+tmpA.x*tmpB.y;
				}

				__syncthreads();
#endif

				// calculate determinants
				vOP::findVectorProduct(vectors, vectorIndex, myMatrix, scrSpace_GPU, scrSpace2_GPU);
				__syncthreads();

				// write out
				if (vectorIndex==0){
					inputMatrices [ mat_00_index ].x = vectors [ vec_00_scr ].x ;
					inputMatrices [ mat_00_index ].y = vectors [ vec_00_scr ].y ;
				}
#endif
			}






#ifdef __DPREC__

#else
			__global__ void reductionGlobalSum(int vestige, float2 *input, float2 *output, int it){
#endif
				int t_id; 

				if (it == 0){
					// first load every second element
					t_id = 2*loadSize_GPU * matSide_GPU* (RED_SUM_SIZE * blockIdx.x + threadIdx.x);
				}else{
					t_id = RED_SUM_SIZE * blockIdx.x + threadIdx.x;

				}
				__shared__ volatile float2 scratch[RED_SUM_SIZE];

				// init
				scratch[threadIdx.x].x = 0.0f;
				scratch[threadIdx.x].y = 0.0f;

				__syncthreads();

				// load
				if ( blockIdx.x != (gridDim.x-1) ){
					scratch[ threadIdx.x ].x = input[ t_id ].x;
					scratch[ threadIdx.x ].y = input[ t_id ].y;
				}
				else{

					if (threadIdx.x < vestige){
						scratch[ threadIdx.x ].x = input[ t_id ].x;
						scratch[ threadIdx.x ].y = input[ t_id ].y;
					}
				}

				__syncthreads();

#if RED_SUM_SIZE == 1024
				if (threadIdx.x < 512){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 512 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 512 ].y;
				}
				__syncthreads();
#endif


#if RED_SUM_SIZE >= 512
				if (threadIdx.x < 256){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 256 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 256 ].y;
				}
				__syncthreads();
#endif


#if RED_SUM_SIZE >= 256
				if (threadIdx.x < 128){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 128 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 128 ].y;
				}
				__syncthreads();
#endif


#if RED_SUM_SIZE >= 128
				if (threadIdx.x < 64){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 64 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 64 ].y;
				}
				__syncthreads();
#endif


#if RED_SUM_SIZE >= 64
				if (threadIdx.x < 32){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 32 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 32 ].y;
				}
#endif


#if RED_SUM_SIZE >= 32
				if (threadIdx.x < 16){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x+16 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x+16 ].y;
				}
#endif

				if (threadIdx.x < 8){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 8 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 8 ].y;
				}

				if (threadIdx.x < 4){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 4 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 4 ].y;
				}

				if (threadIdx.x < 2){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 2 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 2 ].y;
				}

				if (threadIdx.x < 1){
					scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 1 ].x;
					scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 1 ].y;
				}


				if (threadIdx.x == 0){
					output[0].x 	= scratch[0].x;
					output[0].y 	= scratch[0].y;
				}
			}
		}



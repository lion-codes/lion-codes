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
#include <omp.h>
#include <cuda.h>

#include "LUCoreCPU.h"
#include "complex.cuh"
#include "Node.h"
#include "RunInfo.h"

using namespace std;

namespace LUCoreCPU {
	/*
	 * Namespace LUCoreCPU
	 * Function: CALL
	 * Description: performs a batch LU decomposition on "num" "matrices" then calculates the determinants of each of them. Finally adds up the determinant two by two and outputs the result.
	 */
	//#ifdef __DPREC__
	//	void CALL(double2 *matrices, int num, RunInfo *rInfo, int rank, int th_num, double2 *dets)
	//#else
	//#endif

	void CALL(float2 *matrices,float2 *result, int num, RunInfo *rInfo, int rank, int cpuCnt){


		int th_num;

#ifdef __DPREC__
		double2 det;
		double2 sum;

		sum.x = 0.0;
		sum.y = 0.0;
#else
		float2 sum,det1,det2,tmp;
		float2 *common	= new float2[cpuCnt];

		det1.x = 0.0f;
		det1.y = 0.0f;

		det2.x = 0.0f;
		det2.y = 0.0f;

		tmp.x = 0.0f;
		tmp.y = 0.0f;

		sum.x = 0.0f;
		sum.y = 0.0f;
#endif
		int matSide = rInfo->getMatSide();
		int loadSize = rInfo->getLoadSize();

#pragma omp parallel private(det1,det2,tmp,th_num) default(shared)
		{
			th_num= omp_get_thread_num();
#ifdef __DPREC__
			common[th_num].x = 0.0;
			common[th_num].y = 0.0;

#else
			common[th_num].x = 0.0f;
			common[th_num].y = 0.0f;
#endif

			// Parallel for loop; don't wait for other threads to proceed further
#pragma omp for nowait
			for (int i=0; i < num / 2; i++){

				determinant ( &matrices [ 2*i*matSide*loadSize ],&det1,matSide, loadSize);
				determinant ( &matrices [ (2*i+1)*matSide*loadSize ], &det2, matSide, loadSize);

				tmp.x = det1.x*det2.x - det1.y*det2.y;
				tmp.y = det1.y*det2.x + det1.x*det2.y;

				common[th_num].x = common[th_num].x+tmp.x;
				common[th_num].y = common[th_num].y+tmp.y;
			}
#pragma omp critical // Update sum atomically
			{
				sum.x = common[th_num].x + sum.x;
				sum.y = common[th_num].y + sum.y;
			}	

		} // END OF PARALLEL CONSTRUCT	

		// Write the results back
		(*result).x = sum.x;
		(*result).y = sum.y;

		delete [] common;
	}

	//#ifdef __DPREC__
	//	double2 determinant(double2 * inputMatrix, int matSide, int loadSize){
	//#else
	void determinant(float2 * inputMatrix,float2 *det, int matSide, int loadSize){
		//#endif
		// Adapted from numerical recipes in C
		int imax;

#ifdef __DPREC__
		double 		temp,big,d=1,dum;
		double 		den = 0.0;
		double2 	sum, tmp, det;
		sum.x = 1.0;
		sum.y = 0.0;
		tmp.x = 1.0;
		tmp.y = 0.0;
		det.x = 1.0;
		det.y = 0.0;

		double * holder = new double [matSide];
		int idx = 0;
#else
		float temp,big,d=1,dum;
		float den = 0.0f;
		float2 	sum, tmp;
		sum.x = 1.0f;
		sum.y = 0.0f;
		tmp.x = 1.0f;
		tmp.y = 0.0f;
		(*det).x = 1.0f;
		(*det).y = 0.0f;
		float * holder = new float [matSide];
		int idx,idx1,idx2 = 0;
#endif

		// Loop over rows to get scaling info
		for ( int i = 0; i < matSide; i++ ) {
#ifdef __DPREC__
			big=0.0;
#else
			big=0.0f;
#endif
			for (int j = 0; j < matSide; j++){

				temp = fabs(inputMatrix[ i*loadSize+j ].x );
				if ( temp > big )
					big = temp;
			}

			if ( big < 1e-24 ) exit(0); //Singular Matrix/zero entries
			holder[i] = big;
		}

		// Outer loop
		for ( int j=0; j < matSide; j++ ){
			for ( int i=0; i < j; i++ ){
				idx = i*loadSize + j;
				sum.x = inputMatrix[ idx ].x;
				sum.y = inputMatrix[ idx ].y;

				for ( int k = 0; k < i; k++ ) {
					idx1 = i*loadSize + k;
					idx2 = k*loadSize + j;

					tmp.x = inputMatrix[ idx1 ].x * inputMatrix[ idx2 ].x - inputMatrix[ idx1 ].y*inputMatrix[ idx2 ].y;	
					tmp.y = inputMatrix[ idx1 ].y * inputMatrix[ idx2 ].x + inputMatrix[ idx1 ].x*inputMatrix[ idx2 ].y;	

					sum.x = sum.x - tmp.x;
					sum.y = sum.y - tmp.y;
				}	

				inputMatrix[ idx ].x = sum.x;
				inputMatrix[ idx ].y = sum.y;
			}
#ifdef __DPREC__
			big=0.0;
#else
			big=0.0f;
#endif
			// Search for largest pivot
			for ( int i = j; i < matSide; i++) {
				idx = i*loadSize + j;
				sum.x = inputMatrix[ idx ].x;
				sum.y = inputMatrix[ idx ].y;

				for ( int k = 0; k < j; k++ ) {
					idx1 = i*loadSize + k;
					idx2 = k*loadSize + j;

					tmp.x = inputMatrix[ idx1 ].x * inputMatrix[ idx2 ].x - inputMatrix[ idx1 ].y*inputMatrix[ idx2 ].y;	
					tmp.y = inputMatrix[ idx1 ].y * inputMatrix[ idx2 ].x + inputMatrix[ idx1 ].x*inputMatrix[ idx2 ].y;	

					sum.x = sum.x - tmp.x;
					sum.y = sum.y - tmp.y;
				}	
				inputMatrix[ idx ].x = sum.x;
				inputMatrix[ idx ].y = sum.y;

				dum = fabs(sum.x)/ holder[i];

				if (dum >= big) {
					big = dum;
					imax = i;
				}
			}

			// Possible row swap?
			if (j != imax) {
				for ( int k = 0; k < matSide; k++ ) {
					idx1 = imax*loadSize+k;
					idx2 = j*loadSize+k;

					sum.x = inputMatrix[ idx1 ].x;
					sum.y = inputMatrix[ idx1 ].y;

					inputMatrix[ idx1 ].x = inputMatrix[ idx2 ].x;
					inputMatrix[ idx1 ].y = inputMatrix[ idx2 ].y;

					inputMatrix[ idx2 ].x = sum.x;
					inputMatrix[ idx2 ].y = sum.y;
				}
				d = -d;
				holder[imax] = holder[j];
			}

			if (j != matSide-1) {
				idx1 = j*loadSize + j;

				sum.x = inputMatrix[ idx1 ].x;
				sum.y = inputMatrix[ idx1 ].y;

				for ( int i = j+1; i < matSide; i++ ) {
					idx2 = i*loadSize + j;

					den = sum.x*sum.x + sum.y*sum.y;

					tmp.x = (inputMatrix[ idx2 ].x*sum.x + inputMatrix[ idx2 ].y*sum.y)/den; 
					tmp.y = (inputMatrix[ idx2 ].y*sum.x - inputMatrix[ idx2 ].x*sum.y)/den; 

					inputMatrix[ idx2 ].x = tmp.x;
					inputMatrix[ idx2 ].y = tmp.y;
				}
			}

		}

		// actual determinant evaluation
		for(int i=0; i<matSide; i++) {
			idx = i*loadSize + i;
			tmp.x = (*det).x * inputMatrix[ idx ].x - (*det).y * inputMatrix[ idx ].y; 
			tmp.y = (*det).y * inputMatrix[ idx ].x + (*det).x * inputMatrix[ idx ].y; 

			(*det).x = tmp.x;
			(*det).y = tmp.y;
		}
		(*det).x = d*(*det).x;
		(*det).y = d*(*det).y;
		delete [] holder;
	}

}

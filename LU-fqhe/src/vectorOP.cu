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

#include <cuda.h>

#include "vectorOP.h"
#include "LUCoreGPU.h"


namespace vOP {
	__inline__ __device__ bool geq(float A, int indexA, float B, int indexB){

		if (A > B) return 1;
		if ((A == B) && (indexB < indexA)) return 1;
		return 0;

	}

	/*
	 * Find Vector Maxima
	 */
	__inline__ __device__ void findVectorMaxima(volatile float2 * input, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU){

		int DIFF = scrSpace_GPU - scrSpace2_GPU;
		int UB  = scrSpace2_GPU;
		int LB = scrSpace2_GPU-DIFF;

		// calculate the vestige
		if ( ( vectorIndex < UB ) && ( vectorIndex >= LB ) )
			if ( input[ vec_index + DIFF ].x > input[ vec_index ].x ){
				input[ vec_index ].x = input[ vec_index + DIFF ].x;
			}
		__syncthreads();

		if( scrSpace2_GPU == 512 ) {
			if (vectorIndex < 256)
				if ( input[vec_index + 256 ].x > input[ vec_index ].x ){
					input[ vec_index ].x = input[ vec_index + 256 ].x;
				}
		}
		__syncthreads();



		if( scrSpace2_GPU >= 256 ) {
			if (vectorIndex < 128)
				if ( input[ vec_index + 128 ].x > input[vec_index ].x ){
					input[vec_index].x = input[vec_index + 128].x;
				}
		}
		__syncthreads();


		if( scrSpace2_GPU >= 128 ) {
			if (vectorIndex < 64)
				if ( input[vec_index + 64].x > input[vec_index].x ){
					input[vec_index].x = input[vec_index + 64].x;
				}
		}
		__syncthreads();

		if( scrSpace2_GPU >= 64 ) {
			if (vectorIndex < 32)
				if ( input[vec_index + 32].x > input[vec_index].x ){
					input[vec_index].x = input[vec_index + 32].x;
				}
		}
		__syncthreads();

		if( scrSpace2_GPU >= 32 ) {
			if (vectorIndex < 16)
				if ( input[vec_index + 16].x > input[vec_index].x ){
					input[vec_index].x = input[vec_index + 16].x;
				}
		}

		if( scrSpace2_GPU >= 16 ) {
			if (vectorIndex < 8)
				if ( input[vec_index + 8].x > input[vec_index].x ){
					input[vec_index].x = input[vec_index + 8].x;
				}
		}

		if( scrSpace2_GPU >= 8 ) {
		if (vectorIndex < 4)
			if ( input[ vec_index + 4 ].x  > input[ vec_index ].x ){
				input[ vec_index ].x = input[ vec_index + 4 ].x;
			}
		}

		if (vectorIndex < 2)
			if ( input[ vec_index + 2 ].x > input[ vec_index ].x ){
				input[ vec_index ].x = input[ vec_index + 2 ].x;
			}

		if (vectorIndex < 1)
			if ( input[ vec_index + 1 ].x > input[ vec_index ].x ){
				input[ vec_index ].x = input[ vec_index + 1 ].x;
			}
	}

	/*
	 * Find Vector Maxima Key
	 */
	__inline__ __device__ void findVectorMaximaKey( volatile float2 * input, volatile int * indices, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU){
		int DIFF = scrSpace_GPU - scrSpace2_GPU;
		int UB  = scrSpace2_GPU;
		int LB = scrSpace2_GPU-DIFF;
		// calculate the vestige
		if (( vectorIndex < UB ) && ( vectorIndex >= LB ))

			if ( geq( input [ vec_index + DIFF ].x , indices [ vec_index +DIFF ],input [ vec_index ].x , indices [ vec_index ] )){

				input [ vec_index ].x = input [ vec_index + DIFF ].x;
				input [ vec_index ].y = input [ vec_index + DIFF ].y;
				indices [ vec_index ] = indices [ vec_index + DIFF ];

			}
		__syncthreads();


		if( scrSpace2_GPU == 512 ) {
			if ( vectorIndex < 256 )

				if ( geq( input [ vec_index + 256 ].x , indices [ vec_index +256 ],input [ vec_index ].x , indices[ vec_index ])){

					input [ vec_index ].x = input [ vec_index + 256 ].x;
					input [ vec_index ].y = input [ vec_index + 256 ].y;
					indices [ vec_index ] = indices [ vec_index + 256 ];

				}
		}
		__syncthreads();


		if( scrSpace2_GPU >= 256 ) {
			if (vectorIndex < 128)

				if ( geq( input [ vec_index + 128 ].x , indices [ vec_index +128 ],input [ vec_index ].x , indices[ vec_index ])){

					input [ vec_index ].x = input [ vec_index + 128 ].x;
					input [ vec_index ].y = input [ vec_index + 128 ].y;
					indices [ vec_index ] = indices [ vec_index + 128 ];
				}
		}
		__syncthreads();



		if( scrSpace2_GPU >= 128 ) {
			if (vectorIndex < 64)

				if ( geq( input [ vec_index + 64 ].x , indices [ vec_index +64 ],input [ vec_index ].x , indices[ vec_index ])){

					input[vec_index].x = input[vec_index + 64].x;
					input[vec_index].y = input[vec_index + 64].y;
					indices [ vec_index ] = indices [ vec_index + 64 ];
				}
		}
		__syncthreads();



		if( scrSpace2_GPU >= 64 ) {
			if (vectorIndex < 32)

				if ( geq( input [ vec_index + 32 ].x, indices [ vec_index +32 ],input [ vec_index ].x, indices[ vec_index ])){

					input [ vec_index ].x = input [ vec_index + 32 ].x;
					input [ vec_index ].y = input [ vec_index + 32 ].y;
					indices [ vec_index ] = indices [ vec_index + 32 ];
				}
		}
		__syncthreads();


		if( scrSpace2_GPU >= 32 ) {
			if (vectorIndex < 16)

				if ( geq( input [ vec_index + 16 ].x, indices [ vec_index +16 ],input [ vec_index ].x, indices[ vec_index ])){

					input [ vec_index ].x = input [ vec_index + 16 ].x;
					input [ vec_index ].y = input [ vec_index + 16 ].y;
					indices [ vec_index ] = indices [ vec_index + 16 ];
				}
		}

		if( scrSpace2_GPU >= 16 ) {
			if (vectorIndex < 8)

				if ( geq( input [ vec_index + 8 ].x, indices [ vec_index +8 ],input [ vec_index ].x, indices[ vec_index ])){


					input [ vec_index ].x = input [ vec_index + 8 ].x;
					input [ vec_index ].y = input [ vec_index + 8 ].y;
					indices [ vec_index ] = indices [ vec_index + 8 ];
				}
		}


		if( scrSpace2_GPU >= 8 ) {
		if (vectorIndex < 4)

			if ( geq( input [ vec_index + 4 ].x , indices [ vec_index +4 ],input [ vec_index ].x , indices[ vec_index] )){

				input [ vec_index ].x = input [ vec_index + 4 ].x;
				input [ vec_index ].y = input [ vec_index + 4 ].y;
				indices [ vec_index ] = indices [ vec_index + 4 ];
			}
		}

		if (vectorIndex < 2)

			if ( geq( input [ vec_index + 2].x, indices [ vec_index +2 ],input [ vec_index ].x, indices[ vec_index ])){

				input [ vec_index ].x = input [ vec_index + 2 ].x;
				input [ vec_index ].y = input [ vec_index + 2 ].y;
				indices [ vec_index ] = indices [ vec_index + 2 ];

			}
		if (vectorIndex < 1)

			if ( geq( input [ vec_index + 1].x, indices [ vec_index +1 ],input [ vec_index ].x, indices[ vec_index ])){

				input [ vec_index ].x = input [ vec_index + 1 ].x;
				input [ vec_index ].y = input [ vec_index + 1 ].y;
				indices [ vec_index ] = indices [ vec_index + 1 ];
			}
	}


	/*
	 * Find Vector Product
	 */
	__inline__ __device__ void findVectorProduct(volatile float2 * input, int vectorIndex, int myMatrix, int scrSpace_GPU, int scrSpace2_GPU){
		int DIFF = scrSpace_GPU - scrSpace2_GPU;
		int UB  = scrSpace2_GPU;
		int LB = scrSpace2_GPU-DIFF;

		float2 tmp;
		if(scrSpace_GPU!=scrSpace2_GPU) {
			// calculate the vestige
			if (( vectorIndex < UB ) && ( vectorIndex >= LB )){
				tmp.x 	= input[vec_index].x * input [ vec_index +DIFF ].x - input[vec_index].y * input [ vec_index +DIFF ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +DIFF ].x + input[vec_index].x * input [ vec_index +DIFF ].y;


				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;
			}
		}
		__syncthreads();



		if( scrSpace2_GPU == 512 ) {
			if ( vectorIndex < 256 ){
				tmp.x 	= input[vec_index].x * input [ vec_index +256 ].x - input[vec_index].y * input [ vec_index +256 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +256 ].x + input[vec_index].x * input [ vec_index +256 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;
			}
		}
		__syncthreads();



		if( scrSpace2_GPU >= 256 ) {
			if ( vectorIndex < 128 ){
				tmp.x 	= input[vec_index].x * input [ vec_index +128 ].x - input[vec_index].y * input [ vec_index +128 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +128 ].x + input[vec_index].x * input [ vec_index +128 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;
			}
		}
		__syncthreads();



		if( scrSpace2_GPU >= 128 ) {
			if (vectorIndex < 64){
				tmp.x 	= input[vec_index].x * input [ vec_index +64 ].x - input[vec_index].y * input [ vec_index +64 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +64 ].x + input[vec_index].x * input [ vec_index +64 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;

			}
		}
		__syncthreads();


		if( scrSpace2_GPU >= 64 ) {
			if (vectorIndex < 32){
				tmp.x 	= input[vec_index].x * input [ vec_index +32 ].x - input[vec_index].y * input [ vec_index +32 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +32 ].x + input[vec_index].x * input [ vec_index +32 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;

			}
		}



		if( scrSpace2_GPU >= 32 ) {
			if (vectorIndex < 16){
				tmp.x 	= input[vec_index].x * input [ vec_index +16 ].x - input[vec_index].y * input [ vec_index +16 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +16 ].x + input[vec_index].x * input [ vec_index +16 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;

			}
		}

		if( scrSpace2_GPU >= 16 ) {
			if (vectorIndex < 8){
				tmp.x 	= input[vec_index].x * input [ vec_index +8 ].x - input[vec_index].y * input [ vec_index +8 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +8 ].x + input[vec_index].x * input [ vec_index +8 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;

			}
		}


		if( scrSpace2_GPU >= 8 ) {
			if (vectorIndex < 4){
				tmp.x 	= input[vec_index].x * input [ vec_index +4 ].x - input[vec_index].y * input [ vec_index +4 ].y;
				tmp.y 	= input[vec_index].y * input [ vec_index +4 ].x + input[vec_index].x * input [ vec_index +4 ].y;

				input [ vec_index ].x 	= tmp.x;
				input [ vec_index ].y 	= tmp.y;

			}
		}

		if (vectorIndex < 2){
			tmp.x 	= input[vec_index].x * input [ vec_index +2 ].x - input[vec_index].y * input [ vec_index +2 ].y;
			tmp.y 	= input[vec_index].y * input [ vec_index +2 ].x + input[vec_index].x * input [ vec_index +2 ].y;

			input [ vec_index ].x 	= tmp.x;
			input [ vec_index ].y 	= tmp.y;
		}
		if (vectorIndex < 1){

			tmp.x 	= input[vec_index].x * input [ vec_index +1 ].x - input[vec_index].y * input [ vec_index +1 ].y;
			tmp.y 	= input[vec_index].y * input [ vec_index +1 ].x + input[vec_index].x * input [ vec_index +1 ].y;

			input [ vec_index ].x 	= tmp.x;
			input [ vec_index ].y 	= tmp.y;

		}
	}
}

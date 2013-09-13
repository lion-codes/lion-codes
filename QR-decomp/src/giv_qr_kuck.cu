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
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "main.h"
#include "utilities.h"

__device__ __constant__ int cmem_size_kuck,cmem_size_MM_kuck;

__device__ int delta_k(int k) {
	return ( (k&0x01)==0 ? 1 : 0 );
}	
/* 
 * Description: initializes q_temp to the identity matrix
 *
 * TODO: Optimize and avoid for loop; use one big kernel launch
 */ 
__global__ void init_q(int k, float2 *q_temp) {
	// Buffer rows
	__shared__ float2 upper_row[NMPBL*MATRIX_SIDE];
	__shared__ float2 lower_row[NMPBL*MATRIX_SIDE];

	// index to matrix for processing
	int myMatrix 		= threadIdx.x / MATRIX_SIDE;
	// index to vector for processing
	int vectorIndex 	= threadIdx.x % MATRIX_SIDE;
	// matrix offset for this block
	int memoryStride        = ( blockIdx.x * NMPBL + myMatrix  ) * MATRIX_SIDE * MATRIX_SIDE ;

	// Set column and line number we want to eliminate
	int my_i, my_j = 0;
	int dk = delta_k(k);
	if( k < MATRIX_SIDE ) {
		my_j = blockIdx.z;
		my_i = (MATRIX_SIDE-k) + 2*my_j;
	} else {
		my_j = (k-MATRIX_SIDE) + blockIdx.z + 1;
		my_i = (k-MATRIX_SIDE) + 2*blockIdx.z + 2;
	}	

	__syncthreads();
	int mask = (vectorIndex == my_i);
	
	lower_row[threadIdx.x].x = (float)mask;
	lower_row[threadIdx.x].y = 0.0f;

	mask = (vectorIndex == my_i-1);	
	upper_row[threadIdx.x].x = (float)mask;
	upper_row[threadIdx.x].y = 0.0f;

	__syncthreads();
	q_temp[memoryStride + (my_i-1) * MATRIX_SIDE + vectorIndex ] 	= upper_row[threadIdx.x];
	q_temp[memoryStride + my_i * MATRIX_SIDE + vectorIndex ] 		= lower_row[threadIdx.x];

}


/*
	__shared__ float2 row[NMPBL*MATRIX_SIDE];

	// index to matrix for processing
	int myMatrix 		= threadIdx.x / MATRIX_SIDE;

	// index to vector for processing
	int vectorIndex 	= threadIdx.x % MATRIX_SIDE;

	// matrix offset for this block
	int memoryStride        = ( blockIdx.x * NMPBL + myMatrix  ) * MATRIX_SIDE * MATRIX_SIDE ;

	for(int i = 0;i<MATRIX_SIDE;i++) {

		row[threadIdx.x].x = (float)diag_mask;
		row[threadIdx.x].y = 0.0f;
		
		if(memoryStride + i*MATRIX_SIDE + vectorIndex < cmem_size_kuck*MATRIX_SIDE*MATRIX_SIDE) {		
			q_temp[memoryStride + i*MATRIX_SIDE + vectorIndex].x = row[threadIdx.x].x;
			q_temp[memoryStride + i*MATRIX_SIDE + vectorIndex].y = row[threadIdx.x].y;
		}
	}	
	__syncthreads();
}	
*/


/* Kernel overview
 * 
 * One block processes multiple complex matrices, performing more than one Givens rotation 
 * at a time.
 * One ensemble of threads processes one line of a matrix at a time.
 *
 * See 
 * - A.H. Sameh and D. J. Kuck, "On stable parallel linear system solvers"
 * - M. Cosnard, Y. Robert, "Complexity of parallel QR factorization"
 *
 * PYT 07/26
 */
__global__ void qr_gpu_kuck_bat(int k,float2 *matrices, float2 *q_temp,float2 *q_complete) {

	// Buffer rows
	__shared__ float2 upper_row[NMPBL*MATRIX_SIDE];
	__shared__ float2 lower_row[NMPBL*MATRIX_SIDE];

	__shared__ float2 upper_Q[NMPBL*MATRIX_SIDE];
	__shared__ float2 lower_Q[NMPBL*MATRIX_SIDE];
	// index to matrix for processing
	int myMatrix 		= threadIdx.x / MATRIX_SIDE;
	// index to vector for processing
	int vectorIndex 	= threadIdx.x % MATRIX_SIDE;
	// matrix offset for this block
	int memoryStride        = ( blockIdx.x * NMPBL + myMatrix  ) * MATRIX_SIDE * MATRIX_SIDE ;

	// Set column and line number we want to eliminate
	int my_i, my_j = 0;
	int dk = delta_k(k);
	if( k < MATRIX_SIDE ) {
		my_j = blockIdx.z;
		my_i = (MATRIX_SIDE-k) + 2*my_j;
	} else {
		my_j = (k-MATRIX_SIDE) + blockIdx.z + 1;
		my_i = (k-MATRIX_SIDE) + 2*blockIdx.z + 2;
	}	

	// Load row data
	if(memoryStride + my_i*MATRIX_SIDE + vectorIndex < cmem_size_kuck*MATRIX_SIDE*MATRIX_SIDE) {
		upper_row[threadIdx.x] = matrices[memoryStride + (my_i-1)*MATRIX_SIDE + vectorIndex]; // Upper row
		lower_row[threadIdx.x] = matrices[memoryStride + my_i*MATRIX_SIDE + vectorIndex]; // Lower row w/ leading zero
	}	

	// Calculate c and s
	float2 u,v,c,s;
	float f,g,den;

	__syncthreads();
	u = upper_row[myMatrix*MATRIX_SIDE + my_j]; // broadcast operation from SMEM
	v = lower_row[myMatrix*MATRIX_SIDE + my_j]; // broadcast operation from SMEM

	f = u.x*u.x + u.y*u.y;
	g = v.x*v.x + v.y*v.y;

	if( g < 2e-16 ) {
		c.x = 1.0f;
		c.y = 0.0f;

		s.x = 0.0f;
		s.y = 0.0f;

	} else if (f< 2e-16) {
		c.x = 0.0f;
		c.y = 0.0f;

		// s = conj(v)/g
		den = 1.0f/g;
		s.x = v.x*den; 
		s.y = -v.y*den;
	} else {
		// r = sqrt(f + g)
		den = rsqrt(f + g);
		// c = f/r
		c.x = sqrt(f)*den;
		c.y = 0.0f;

		// s = x/f * conj(y) / r
		// den = -1/(f*r)
		den *= rsqrt(f);
		
		s.x = (u.x*v.x + u.y*v.y)*den;
		s.y = (u.y*v.x - u.x*v.y)*den;
	}	

	//// Compute the two rows update
	// u*c + v*s
	// Load data 
	__syncthreads();
	u = upper_row[threadIdx.x];
	v = lower_row[threadIdx.x];
	// Perform product: real part 
	f = (u.x*c.x - u.y*c.y) + (v.x*s.x - v.y*s.y); 
	// Perform product: imaginary part
	g = (u.x*c.y + u.y*c.x) + (v.x*s.y + v.y*s.x);

	float2 tmp;
	tmp.x = f;
	tmp.y = g;

	// u*-conj(s) + v*c
	// Perform product: real part 
	f = -(u.x*s.x + u.y*s.y) + (v.x*c.x - v.y*c.y); 
	// Perform product: imaginary part
	g = (u.x*s.y - u.y*s.x) + (v.x*c.y + v.y*c.x);

	// Store
	__syncthreads();
	upper_row[threadIdx.x] = tmp;

	lower_row[threadIdx.x].x = f;
	lower_row[threadIdx.x].y = g;

	// Write data in the original matrix
	if(memoryStride + my_i*MATRIX_SIDE + vectorIndex < cmem_size_kuck*MATRIX_SIDE*MATRIX_SIDE) {
		matrices[memoryStride + (my_i-1) * MATRIX_SIDE + vectorIndex ] = upper_row[threadIdx.x];

		matrices[memoryStride + my_i * MATRIX_SIDE + vectorIndex ].x     = (vectorIndex > my_j) * lower_row[threadIdx.x].x;
		matrices[memoryStride + my_i * MATRIX_SIDE + vectorIndex ].y     = (vectorIndex > my_j) * lower_row[threadIdx.x].y;
	}

	// Write to q_temp
	/*
	__syncthreads();
	int mask = (vectorIndex == my_i || vectorIndex == (my_i-1));
	upper_row[threadIdx.x].x = mask*( (vectorIndex == (my_i-1))*c.x + (vectorIndex == (my_i))*s.x );
	upper_row[threadIdx.x].y = mask*( (vectorIndex == (my_i-1))*c.y + (vectorIndex == (my_i))*s.y );

	lower_row[threadIdx.x].x = mask*( (vectorIndex == -(my_i-1))*s.x + (vectorIndex == (my_i))*c.x );
	lower_row[threadIdx.x].y = mask*( (vectorIndex == (my_i-1))*s.y + (vectorIndex == (my_i))*c.y );
*/
	// Load from q_complete: two rows are only necessary	
	__syncthreads();
	upper_Q[threadIdx.x] = q_complete[memoryStride + (my_i-1) * MATRIX_SIDE + vectorIndex ]; 	
	lower_Q[threadIdx.x] = q_complete[memoryStride + my_i * MATRIX_SIDE + vectorIndex ];

	// Perform the multiplication: QCOMPLETE = QTEMP * QCOMPLETE
	u = upper_Q[threadIdx.x]; 
	v = lower_Q[threadIdx.x];

	// Q[i-1,k] = C*Q[i-1,k] + S*Q[i,k]
	upper_row[threadIdx.x].x = (u.x*c.x - u.y*c.y) + (v.x*s.x - v.y*s.y); 
	upper_row[threadIdx.x].y = (u.x*c.y + u.y*c.x) + (v.x*s.y + v.y*s.x);

	// Q[i,k] = -S'*Q[i-1,k] + C*Q[i,k]
	lower_row[threadIdx.x].x = -(u.x*s.x + u.y*s.y) + (v.x*c.x - v.y*c.y); 
	lower_row[threadIdx.x].y =  (u.x*s.y - u.y*s.x) + (v.x*c.y + v.y*c.x);

	__syncthreads();
	// Write to global
	if(memoryStride + my_i*MATRIX_SIDE + vectorIndex < cmem_size_kuck*MATRIX_SIDE*MATRIX_SIDE) {
		q_complete[memoryStride + (my_i-1) * MATRIX_SIDE + vectorIndex ] 	= upper_row[threadIdx.x];
		q_complete[memoryStride + my_i * MATRIX_SIDE + vectorIndex ] 		= lower_row[threadIdx.x];
	}
}



extern "C"{
	void givens_qr_kuck_bat(float * mats, int size, float * q){

		float2 *q_tempA,*q_tempB,*q_complete,*matrices;

		//initialize Q
		for (int k=0; k<size; k++)
			for (int i=0; i<MATRIX_SIDE; i++)		
				for (int j=0; j<2*MATRIX_SIDE; j++)
					q[j+i*2*MATRIX_SIDE+k*MATRIX_SIDE*MATRIX_SIDE*2] = (2*i==j) ? 1.0 :0.0;		

		int qsize = size*MATRIX_SIDE*MATRIX_SIDE;
		cudaMemcpyToSymbol(cmem_size_kuck,&size,sizeof(size));
		cudaMemcpyToSymbol(cmem_size_MM_kuck,&qsize,sizeof(qsize));

		// Allocate memory and copy data
		cudaMalloc((void**) &q_complete, sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);
		cudaMalloc((void**) &q_tempA, sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);
		cudaMalloc((void**) &q_tempB, sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);

		cudaMemcpy(q_complete, q, sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*size, cudaMemcpyHostToDevice);
		cudaMemcpy(q_tempA, q, sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*size, cudaMemcpyHostToDevice);

		cudaMalloc((void**) &matrices, sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);
		cudaMemcpy(matrices, mats, sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*size, cudaMemcpyHostToDevice);

		int nrows = MATRIX_SIDE;
		int ncol = MATRIX_SIDE;

		dim3 threads,blocks;

		threads.x 	= MATRIX_SIDE;
		blocks.x	= (int)ceil((float)size/(float)NMPBL);
		int dim1d_copy2tmp = (int)ceil(sqrt((float)size*(float)MATRIX_SIDE*(float)MATRIX_SIDE/(float)NTH));
		dim3 grid_copy2tmp;
		grid_copy2tmp.x = dim1d_copy2tmp;
		grid_copy2tmp.y = dim1d_copy2tmp;

		double begin = omp_get_wtime();	
		for(int k = 1;k<=2*MATRIX_SIDE-3;k++) {

			if( k < MATRIX_SIDE ) {
				blocks.z = (int)ceil((float)k/2.0f);
			} else {
				blocks.z = (int)ceil((float)k/2.0f) - k + MATRIX_SIDE-1;
			}	

			// Launch the main kernel; calculates the Givens rotations and places them in the temporary matrix Q_A
			qr_gpu_kuck_bat <<< blocks,NTH >>> (k,matrices,q_tempA,q_complete); 
			#ifdef GPU_DEBUG
			checkCUDAError("QR_GPU_BAT FAILED",__FILE__,__LINE__-1);
			#endif
		}//end main loops



	#ifdef _QR_VERBOSE_

		float2 * q_dump = (float2 *) malloc(sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);
		cudaMemcpy(q_dump, q_complete, sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*size, cudaMemcpyDeviceToHost);
		float2 * r_dump = (float2 *) malloc(sizeof(float2)*size*MATRIX_SIDE*MATRIX_SIDE);
		cudaMemcpy(r_dump, matrices, sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*size, cudaMemcpyDeviceToHost);


		// Take the transpose of q_complete is CUBLAS
		for (int k=0; k<size; k++){
			printf("q %i\n",k);
			for (int j=0; j<MATRIX_SIDE; j++) {
				for (int i=0; i<MATRIX_SIDE; i++){
					printf("%f+%fi, ",q_dump[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].x,
							q_dump[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].y);
			
				}	
					printf(";\n");
			}	

		}
		for (int k=0; k<size; k++){
			printf("r_mat %i\n",k);
			for (int i=0; i<MATRIX_SIDE; i++){
				for (int j=0; j<MATRIX_SIDE; j++)
					printf("%f+%fi, ",r_dump[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].x,
							r_dump[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].y);


				printf(";\n",i);
			}

		}


		free(r_dump);
		free(q_dump);

	#endif

		cudaDeviceSynchronize();
		double end = omp_get_wtime();
		printf("QR KUCK-SAMEH\t %f\n",end-begin);

		cudaFree(q_tempA);
		cudaFree(q_tempB);
		cudaFree(q_complete);
		cudaFree(matrices);
	}
}

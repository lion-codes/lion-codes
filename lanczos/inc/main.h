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


#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>

#ifndef _MAIN_H
#define _MAIN_H

#define _DEBUG_LANCZOS
#define _MAX_SIZE 1024*1024
#define _USE_GPU
#define _CALC_EVECS

//#define _GATHER_SCALAR


#ifdef _USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuComplex.h>


#endif

#define _LAN_THREADS 16

// tolerance in dot products of subsequent e-vectors 
// ie., if norm is greater than this, orthogonalize
#define _EVECS_NORM 0.00001


#ifdef _DEBUG_LANCZOS
#define check_cu_error(x) 	check_last_cuda_error(cerror,x,hostname,__LINE__);
#define	check_cb_error(x)	check_last_cublas_error(status,x,hostname,__LINE__);
#else
#define check_cu_error(x) 	
#define	check_cb_error(x)	
#endif


#define dump_vec(rnk,y,x) printf(y); \
	printf("%i :\n",rnk); for (int i=0; i<n; i++) \
	printf("%f+i%f ",creal(x[i]),cimag(x[i])); printf("\n");

//dump column major, full matrix
#define dump_mat(y,x) printf(y); \
	printf(":\n"); for (int i=0; i<n; i++){ \
	for (int j=0; j<n; j++) printf("%f + %fi ",creal(x[j*n+i]),cimag(x[j*n+i])); printf("\n");}



// blas& lapack defs

void zsteqr_(char * type,//you know
	int * n,		//mat order	
	double * D,		//diagonal
	double * E,		//off-diag
	complex double * evectors,
	int * m,
	double * scratch,	
	int * info);


void dsterf_(int * n, //matrix order
		double * D, 	//diagonal
		double * E, 	//off-diag 
		int * INFO );	//information, of sorts


void zgemv_(char * type,	// transpose or not etc
		int * rows,		// mat rows
		int * cols,		// mat columns
		complex double * mula,	// multiply factor 
		complex double * A,	// matrix
		int * dimn,		// dim
		complex double * x,	// vector on RHS
		int * incx,		// increment in this vector
		complex double * mulb,	// multiply factor
		complex double * y,	// LHS vector
		int * incy);		// increment in this vector

void cgemv_(char * type,	// transpose or not etc
		int * rows,		// mat rows
		int * cols,		// mat columns
		complex float * mula,	// multiply factor 
		complex float * A,	// matrix
		int * dimn,		// dim
		complex float * x,	// vector on RHS
		int * incx,		// increment in this vector
		complex float * mulb,	// multiply factor
		complex float * y,	// LHS vector
		int * incy);		// increment in this vector

void  zgemm_( char * typea, 	// transpose A or not etc
		char * typeb, 		//ditto for B
		int * rows, 		// rows of op(A) and C
		int * cols, 		// cols of op(A) and rows of op(B)
		int * k,		// cols of op(A) and rows of op(B) 
		complex double * mula, 	// constant alpha
		complex double * A, 	// matrix A
		int * dima,		// dim
		complex double * B,	// matrix B
		int * dimb,		// dim
		complex double * mulb, 	// constant beta
		complex double * c, 	// unused
		int * dimc);		// dim


// thin wrappers around kernels
//
#ifdef _USE_GPU
cudaError_t lanczos_first_update(dim3 blocks, 
		dim3 threads, 
		cuDoubleComplex * d_r, 
		cuDoubleComplex * d_Q, 
		cuDoubleComplex * d_beta, 
		int m,
		int i);

cudaError_t lanczos_second_update(dim3 blocks, 
		dim3 threads, 
		cuDoubleComplex * d_r, 
		cuDoubleComplex * d_Q, 
		cuDoubleComplex * d_beta, 
		int m,
		int i);

cudaError_t lanczos_third_update(dim3 blocks,
		dim3 threads,
		cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_alpha,
		int m,
		int i);

cudaError_t vector_dot(cuDoubleComplex * inputA,
		cuDoubleComplex * inputB,
		cuDoubleComplex * output,
		cuDoubleComplex * result,
		int type,
		int size,
		int offsetA,
		int offsetB,
		int sqr);

#endif

#endif

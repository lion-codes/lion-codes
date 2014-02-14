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

#include <omp.h>
#include <cuda.h>

#include "mkl_lapacke.h"
#include "main.h"

void givens_qr_bat(float * mats, int size, float * q);
void givens_qr_mmpb_bat(float * mats, int size, float * q);
void givens_qr_kuck_bat(float * mats, int size, float * q);
void givens_qr_cpu(float *mats,int size, float *q);

// A harness for testing qr
int main(int argc, char * argv[] ){
	if (argc != 2){
		fprintf(stderr,"USAGE : qr.x <batch_size> \n");
		exit(1);
	}

	// Matrix size
	int batSize 		= atoi(argv[1]);

	// Make a test matrix
	float * A = (float *) malloc(sizeof(float) * MATRIX_SIDE * MATRIX_SIDE * batSize * 2);
	for (int i=0; i<MATRIX_SIDE*MATRIX_SIDE*batSize*2; i++) 
		A[i]=rand() / (float) RAND_MAX;

//	float * tau = (float *) malloc(sizeof(float) * MATRIX_SIDE *MATRIX_SIDE * 2 * batSize);
//	float * tau = (float *) malloc(sizeof(float) *MATRIX_SIDE * 2 * batSize);
	float * q = (float *) malloc(sizeof(float) * MATRIX_SIDE *MATRIX_SIDE * 2*batSize);

	double begin,end;

	// PRINT OUT THE ORIGINAL MATRICES
#ifdef _QR_VERBOSE_ 
	printf("*****************\n");
	printf("Original matrices\n");
	printf("*****************\n");
	for (int k=0; k<batSize; k++){
		printf("matrix %i\n",k);
		for (int i=0; i<MATRIX_SIDE; i++){
			for (int j=0; j<MATRIX_SIDE; j++)
				printf("%f+%fi ",A[2*j+i*MATRIX_SIDE*2 + k*MATRIX_SIDE*MATRIX_SIDE*2],A[2*j+i*MATRIX_SIDE*2+1+k*MATRIX_SIDE*MATRIX_SIDE*2]);

			printf(";\n");
		}

	}
#endif
	
	// Launch the original QR function
#ifdef _QR_VERBOSE_
	printf("\n***********\n");
	printf("GPU routine\n");
	printf("***********\n");
#endif
//	givens_qr_kuck_bat(A,batSize,q);


	/****************
	 * CPU ROUTINE *
	 ****************/
#ifdef _QR_VERBOSE_
	printf("\n***********\n");
	printf("CPU routine\n");
	printf("***********\n");
#endif
 	givens_qr_cpu(A,batSize,q);


	/****************
	 * MKL ROUTINE *
	 ****************/
	int m=MATRIX_SIDE,n=MATRIX_SIDE,info,ret_MKL;

#ifdef _QR_VERBOSE_
	printf("\n***********\n");
	printf("MKL routine\n");
	printf("***********\n");
#endif


	// Store data in MKL_Complex8
	MKL_Complex8 * MKL_A = (MKL_Complex8 *) malloc(MATRIX_SIDE*MATRIX_SIDE*batSize*sizeof(MKL_Complex8));
	MKL_Complex8 * MKL_TAU = (MKL_Complex8 *) malloc(MATRIX_SIDE*batSize*sizeof(MKL_Complex8));

/*
	for(int k = 0; k < batSize; k++) {
		for(int i = 0; i < MATRIX_SIDE; i++) {
			for(int j = 0; j < MATRIX_SIDE; j++) {
				MKL_A[j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].real = A[2*j+i*MATRIX_SIDE*2 + k*MATRIX_SIDE*MATRIX_SIDE*2];
				MKL_A[j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].imag = A[2*j+i*MATRIX_SIDE*2+1+k*MATRIX_SIDE*MATRIX_SIDE*2]; 
			}	
		}	
	}	


	begin = omp_get_wtime();	
	#pragma omp parallel for
	for (int i=0; i<batSize; i++){
//		ret_MKL = LAPACKE_cgeqrf( LAPACK_ROW_MAJOR, m, n, (MKL_Complex8 *) &A[i*MATRIX_SIDE*MATRIX_SIDE], m, (MKL_Complex8 *) &tau[i*MATRIX_SIDE]);
		ret_MKL = LAPACKE_cgeqrf( LAPACK_ROW_MAJOR, m, n, &MKL_A[i*MATRIX_SIDE*MATRIX_SIDE], m, &MKL_TAU[i*MATRIX_SIDE]);
	}
	end = omp_get_wtime();
	printf("MKL \t %f (%d return)\n",end-begin,ret_MKL);

	// Print the results of the QR decomposition
	for(int k = 0; k < batSize; k++) {
		for(int i = 0; i < MATRIX_SIDE; i++) {
			for(int j = 0; j < MATRIX_SIDE; j++) {
				A[2*j+i*MATRIX_SIDE*2 + k*MATRIX_SIDE*MATRIX_SIDE*2] = MKL_A[j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].real;
				A[2*j+i*MATRIX_SIDE*2+1+k*MATRIX_SIDE*MATRIX_SIDE*2] = MKL_A[j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].imag; 
			}	
		}	
	}	
	
#ifdef _QR_VERBOSE_ 
	for (int k=0; k<batSize; k++){
		printf("MKL matrix %i\n",k);
		for (int i=0; i<MATRIX_SIDE; i++){
			for (int j=0; j<MATRIX_SIDE; j++)
				printf("%f+%fi ",A[2*j+i*MATRIX_SIDE*2 + k*MATRIX_SIDE*MATRIX_SIDE*2],A[2*j+i*MATRIX_SIDE*2+1+k*MATRIX_SIDE*MATRIX_SIDE*2]);

			printf(";\n");
		}
	}
#endif
*/
	free(A);
	//free(tau);
	free(MKL_A);
	free(MKL_TAU);
	free(q);
	return 0;

}

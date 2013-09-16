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
#include <cuComplex.h>
#include <stdio.h>
#include "lan.h"

__device__ void reduce(volatile cuDoubleComplex * scratch, int size){



#if _DOT_THREADS == 1024
	if ((threadIdx.x < 512) && (size ==1024)){ 
		// unsupported	by compiler
		//scratch [ threadIdx.x ] 	= cuCadd(scratch [ threadIdx.x ] , scratch [ threadIdx.x + 512] );
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 512].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 512].y;
	}
	__syncthreads();
#endif


#if _DOT_THREADS >= 512
	if ((threadIdx.x < 256) && (size>=512)){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 256].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 256].y;
	}
	__syncthreads();
#endif


#if _DOT_THREADS >= 256
	if ((threadIdx.x < 128) && (size>=256)){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 128].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 128].y;
	}
	__syncthreads();
#endif


#if _DOT_THREADS >= 128
	if ((threadIdx.x < 64) && (size>=128)){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 64].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 64].y;
	}
	__syncthreads();
#endif


#if _DOT_THREADS >= 64
	if ((threadIdx.x < 32) && (size>=64)){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 32].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 32].y;
	}
#endif


#if _DOT_THREADS >= 32
	if ((threadIdx.x < 16) && (size>=32)){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 16].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 16].y;
	}
#endif	

	if (threadIdx.x < 8){
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 8].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 8].y;
	}

	if (threadIdx.x < 4){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 4].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 4].y;
	}

	if (threadIdx.x < 2){
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 2].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 2].y;
	}

	if (threadIdx.x < 1){ 	
		scratch [ threadIdx.x ].x 	+= scratch [ threadIdx.x + 1].x;
		scratch [ threadIdx.x ].y 	+= scratch [ threadIdx.x + 1].y;

	}



} 


__global__ void vector_dot_single(cuDoubleComplex * inputA, 
		cuDoubleComplex * inputB, 
		cuDoubleComplex * result, 
		int type,
		int size,
		int offsetA,
		int offsetB,
		int sqr){

	__shared__ volatile cuDoubleComplex scratch[_DOT_THREADS];


	cuDoubleComplex tmpA,tmpB;

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j>=size) return;



	if (threadIdx.x < size){
		tmpA.x = inputA [j+offsetA].x;
		tmpA.y = (type) ? -inputA [j+offsetA].y : inputA [j+offsetA].y;
		tmpB.x = inputB [j+offsetB].x;
		tmpB.y = inputB [j+offsetB].y;

		scratch [ threadIdx.x ].x = tmpA.x*tmpB.x-tmpA.y*tmpB.y;
		scratch [ threadIdx.x ].y = tmpA.y*tmpB.x+tmpA.x*tmpB.y;

	}

	__syncthreads();

	reduce(scratch,size);

	__syncthreads();
	if (threadIdx.x == 0) { 

		if (sqr){
			tmpA.x = sqrt(scratch[0].x);
			tmpA.y = sqrt(scratch[0].y);

		} else{
			tmpA.x = scratch[0].x;
			tmpA.y = scratch[0].y;

		}

		result[threadIdx.x].x 	= tmpA.x; 
		result[threadIdx.x].y 	= tmpA.y; 
	} 



}



__global__ void vector_dot(cuDoubleComplex * inputA, 
		cuDoubleComplex * inputB, 
		cuDoubleComplex * output,
		cuDoubleComplex * result, 
		int type,
		int size,
		int final,
		int offsetA,
		int offsetB,
		int sqr){

	__shared__ volatile cuDoubleComplex scratch[_DOT_THREADS];


	cuDoubleComplex tmpA,tmpB;

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j>=size) return;

	// 
	if (!final){
		// unsupported by compiler <= 5.0
		// scratch[threadIdx.x] = cuCmul(inputA[j],inputB[j]);

		tmpA.x = inputA [j+offsetA].x;
		tmpA.y = (type) ? -inputA [j+offsetA].y : inputA [j+offsetA].y;
		tmpB.x = inputB [j].x;
		tmpB.y = inputB [j].y;


		scratch [ threadIdx.x ].x = tmpA.x*tmpB.x-tmpA.y*tmpB.y;
		scratch [ threadIdx.x ].y = tmpA.y*tmpB.x+tmpA.x*tmpB.y;
	}
	else{
		scratch [ threadIdx.x ].x = output[j].x;
		scratch [ threadIdx.x ].y = output[j].y;
	}	

	__syncthreads();


	reduce(scratch,size);

	__syncthreads();
	if ((threadIdx.x == 0) && !(final)) {  
		output[blockIdx.x].x 	= scratch[0].x; 
		output[blockIdx.x].y 	= scratch[0].y; 
	} 
	if ((threadIdx.x == 0) && (final)) {  
		result[threadIdx.x].x 	= scratch[0].x; 
		result[threadIdx.x].y 	= scratch[0].y; 
	} 



}


void __global__ lanczos_first_update(cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_beta,
		int m,
		int i){

	__shared__ cuDoubleComplex beta;


	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j >= m) return;
	if (threadIdx.x==0) beta = d_beta[i-1];
	__syncthreads();
	d_Q[i*m+j] = cuCdiv(d_r[j], beta);


}
void __global__ lanczos_second_update(cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_beta,
		int m,
		int i){

	__shared__ cuDoubleComplex beta;

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j >= m) return;
	if (threadIdx.x==0) beta = d_beta[i];
	__syncthreads();
	d_r[j] 	= cuCsub(d_r[j],cuCmul(beta,d_Q[(i-1)*m+j]));


}

void __global__ lanczos_third_update(cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_alpha,
		int m,
		int i){

	__shared__ cuDoubleComplex alpha;

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j >= m) return;
	if (threadIdx.x==0) alpha = d_alpha[i];
	__syncthreads();
	d_r[j] 	= cuCsub(d_r[j],cuCmul(alpha,d_Q[i*m+j]));

}

void __global__ lanczos_fourth_update(cuDoubleComplex * d_r,
		cuDoubleComplex * d_output,
		int m){

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j >= m) return;
	d_r[j] 	= cuCsub(d_r[j],d_output[j]);

}

extern "C"{

cudaError_t vector_dot(cuDoubleComplex * inputA,
		cuDoubleComplex * inputB,
		cuDoubleComplex * output,
		cuDoubleComplex * result,
		int type,
		int size,
		int offsetA,
		int offsetB,
		int sqr){


	dim3 blocks,threads;

	threads.x = _DOT_THREADS;
	blocks.x = size / threads.x +1 ;

	cudaGetLastError();

	// FIXME : greater sizes
	if (blocks.x ==1){
		vector_dot_single<<<blocks,threads>>>(inputA, inputB, result, type, size,offsetA,offsetB,sqr);
	}
	else if ((blocks.x < _DOT_THREADS) && (blocks.x > 1)){
		vector_dot<<<blocks,threads>>>(inputA, inputB, output,result, type, size,0,offsetA,offsetB,sqr);
		int newSize 	= blocks.x;
		blocks.x 	= 1;
		vector_dot<<<blocks,threads>>>(inputA, inputB, output,result, type, newSize,1,offsetA,offsetB,sqr);
	} else {

		vector_dot<<<blocks,threads>>>(inputA, inputB, output,result, type, size,0,offsetA,offsetB,sqr);
		int newSize	= blocks.x;	
		blocks.x	= newSize / threads.x;
		vector_dot<<<blocks,threads>>>(inputA, inputB, output,result, type, newSize,0,offsetA,offsetB,sqr);
		newSize	= blocks.x;	
		blocks.x	= 1;
		vector_dot<<<blocks,threads>>>(inputA, inputB, output,result, type, newSize,1,offsetA,offsetB,sqr);
	}


	return cudaGetLastError();
}		

cudaError_t lanczos_fourth_update(dim3 blocks,
		dim3 threads,
		cuDoubleComplex * d_r,
		cuDoubleComplex * d_output,
		int m){


	cudaGetLastError();
	lanczos_fourth_update<<<blocks,threads>>>(d_r,d_output,m);

	return cudaGetLastError();



}


cudaError_t lanczos_third_update(dim3 blocks,
		dim3 threads,
		cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_alpha,
		int m,
		int i){

	cudaGetLastError();
	lanczos_third_update<<<blocks,threads>>>(d_r,d_Q,d_alpha,m,i);

	return cudaGetLastError();
}



cudaError_t lanczos_second_update(dim3 blocks,
		dim3 threads,
		cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_beta,
		int m,
		int i){

	cudaGetLastError();
	lanczos_second_update<<<blocks,threads>>>(d_r,d_Q,d_beta,m,i);

	return cudaGetLastError();
}


cudaError_t lanczos_first_update(dim3 blocks, 
		dim3 threads, 
		cuDoubleComplex * d_r, 
		cuDoubleComplex * d_Q, 
		cuDoubleComplex * d_beta, 
		int m,
		int i){


	cudaGetLastError();
	lanczos_first_update<<<blocks,threads>>>(d_r,d_Q,d_beta,m,i);

	return cudaGetLastError();
}

cudaError_t lanczos_diagnostic(dim3 blocks,
		dim3 threads,
		cuDoubleComplex * d_r,
		cuDoubleComplex * d_Q,
		cuDoubleComplex * d_beta,
		cuDoubleComplex * d_alpha,
		int m,
		int i){

	cudaThreadSynchronize();

	cudaGetLastError();

	cuDoubleComplex * Q_tmp = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * m*(i+1)); 
	cuDoubleComplex * r_tmp = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * m); 
	cuDoubleComplex * a_tmp = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * m); 
	cuDoubleComplex * b_tmp = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * (m-1)); 

	printf("Q: \n");
	cudaMemcpy(Q_tmp,d_Q,(i+1)*m*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	for (int j=0; j<=i; j++){
		for (int k=0; k<m; k++){
			printf("%f+%f ",cuCreal(Q_tmp[j*m+k]),cuCimag(Q_tmp[j*m+k]));
		}
		printf("\n");
	}

	printf("r: \n");
	cudaMemcpy(r_tmp,d_r,m*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	for (int k=0; k<m; k++)
		printf("%f+%f ",cuCreal(r_tmp[k]),cuCimag(r_tmp[k]));
	printf("\n");


	printf("a: \n");
	cudaMemcpy(a_tmp,d_alpha,m*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	for (int k=0; k<m; k++)
		printf("%f+%f ",cuCreal(a_tmp[k]),cuCimag(a_tmp[k]));
	printf("\n");
	printf("b: \n");
	cudaMemcpy(b_tmp,d_beta,(m-1)*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	for (int k=0; k<m-1; k++)
		printf("%f+%f ",cuCreal(b_tmp[k]),cuCimag(b_tmp[k]));



	free(Q_tmp);
	free(r_tmp);
	free(a_tmp);
	free(b_tmp);

	return cudaGetLastError();

}

}

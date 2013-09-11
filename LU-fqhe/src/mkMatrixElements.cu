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
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "mkMatrixElements.h"

#include "complex.cuh"

#if 0

#include "checks.cuh"
#include "const.h"


namespace mkElem {
	// scalars
	__constant__ float 	d_r1, d_r2, d_k1, d_k2;
	__constant__ int  	d_mu, d_nu;

	// lookup tables
	__constant__ float 	deviceA[2*MATRIX_SIDE];
	__constant__ float2 	deviceV[2*MATRIX_SIDE];
	__constant__ float2 	deviceU[2*MATRIX_SIDE];
	__constant__ float2 	threadMaskA[MATRIX_SIDE];
	__constant__ float2 	threadMaskB[MATRIX_SIDE];
	__constant__ float2 	deviceM[2*MATRIX_SIDE*MATRIX_SIDE];

#ifdef __DPREC__

#else
void mkMatrixElements(float * phi,float * theta,
							float k1,float k2,float r1,float r2,int mu,	int nu,	int Q,int myPerms,
								short * devPermutations,void * devMatrices){
#endif


		time_t t = time(0);

		// Copy scalars to constant memory
#ifdef __DPREC__

#else
		cudaMemcpyToSymbol(d_r1,&r1,sizeof(float));
		cudaMemcpyToSymbol(d_r2,&r2,sizeof(float));
		cudaMemcpyToSymbol(d_k1,&k1,sizeof(float));
		cudaMemcpyToSymbol(d_k2,&k2,sizeof(float));
		cudaMemcpyToSymbol(d_mu,&mu,sizeof(int));
		cudaMemcpyToSymbol(d_nu,&nu,sizeof(int));
#endif

		// make some thread masks; these null out data for all but last two columns in M

#ifdef __DPREC__

#else
		float2 tmp0,tmp1;	tmp0.x=0.0f; tmp0.y=0.0f; tmp1.x=1.0f; tmp1.y=1.0f;
		float2 * tmask = (float2*) malloc(sizeof(float2) * MATRIX_SIDE);
#endif


		for (int i=0; i<MATRIX_SIDE; i++) tmask[i] = tmp0;

		tmask[MATRIX_SIDE-2] = tmp1;
		cudaMemcpyToSymbol(threadMaskA,tmask,sizeof(float2)*MATRIX_SIDE);

		tmask[MATRIX_SIDE-2] = tmp0;	tmask[MATRIX_SIDE-1] = tmp1;
		cudaMemcpyToSymbol(threadMaskB,tmask,sizeof(float2)*MATRIX_SIDE);

		// create u,v, copy to device constant memory

		float2 * u = (float2*) malloc(sizeof(float2) * MATRIX_SIDE * 2);
		float2 * v = (float2*) malloc(sizeof(float2) * MATRIX_SIDE * 2);

		for (int i=0; i<2*MATRIX_SIDE; i++){
			u[i].x = cos(theta[i]) * cos(phi[i]);
			u[i].y = cos(theta[i]) * sin(phi[i]);
		}
		cudaMemcpyToSymbol(deviceU,u,sizeof(float2)*MATRIX_SIDE*2);


		for (int i=0; i<2*MATRIX_SIDE; i++){
			v[i].x = sin(theta[i]) * cos(phi[i]);
			v[i].y = -sin(theta[i]) * sin(phi[i]);
		}
		cudaMemcpyToSymbol(deviceV,v,sizeof(float2)*MATRIX_SIDE*2);

		// create the N*(N-1) elements of M, copy this and angles to device

		float2 * m = (float2 *) malloc(2*sizeof(float2) * MATRIX_SIDE * MATRIX_SIDE);
		for (int i=0; i < 2*MATRIX_SIDE*MATRIX_SIDE; i++) m[i] = tmp0;

		setupM_Matrix(theta,phi,u, v, Q, m);

		cudaMemcpyToSymbol(deviceM,m,sizeof(float2)*MATRIX_SIDE*MATRIX_SIDE*2);
		cudaMemcpyToSymbol(deviceA,phi,sizeof(float)*MATRIX_SIDE*2);


		free(tmask);	free(u);	free(v);	free(m);

		dim3 threads, blocks;	threads.x = MAT_THREADS;
		int loops = myPerms / MAX_BLOCKS ;
		blocks.x = (loops ==0) ? myPerms : MAX_BLOCKS;

		if (loops==0){
				fprintf(stderr,"%.24s Launching makeMatrices kernel with %i blocks, %i threads on host %s with rank %i\n",ctime(&t),blocks.x,threads.x,hostname,rank);
				makeMatrices<<<blocks, threads>>>(0,(float2 *) devMatrices,devPermutations);
				cudaDeviceSynchronize();
				cudaCheckError("makeMatrices",hostname,rank);

		} else {
			for (int i=0; i<loops; i++){


				fprintf(stderr,"%.24s Lauching makeMatrices kernel with %i blocks, %i threads on host %s with rank %i\n",ctime(&t),blocks.x,threads.x,hostname,rank);
				makeMatrices<<<blocks, threads>>>(i*MAX_BLOCKS,(float2 *) devMatrices,devPermutations);
				cudaDeviceSynchronize();
				cudaCheckError("makeMatrices",hostname,rank);
			}

			blocks.x = myPerms - (loops * MAX_BLOCKS);
			fprintf(stderr,"%.24s Lauching makeMatrices kernel with %i blocks, %i threads on host %s with rank %i\n",ctime(&t),blocks.x,threads.x,hostname,rank);
			makeMatrices<<<blocks, threads>>>(loops*MAX_BLOCKS,(float2 * )devMatrices,devPermutations);
			cudaDeviceSynchronize();
			cudaCheckError("makeMatrices",hostname,rank);
		}

#ifdef TEST_MAKE_MATRICES

		float2 * testData = (float2*) malloc(sizeof(float2) * MATRIX_SIDE * LOAD_SIZE * myPerms);
		cudaMemcpy(testData, devMatrices, sizeof(float2) * MATRIX_SIDE * LOAD_SIZE * myPerms, cudaMemcpyDeviceToHost);


		int alignMat = MATRIX_SIDE * LOAD_SIZE;

		for (int i=0; i<myPerms; i++){
			fprintf(stderr,"matrix no.%d calculated on device :\n",i);
			for (int j=0; j<MATRIX_SIDE; j++){
				for (int k=0; k<MATRIX_SIDE; k++){
					fprintf(stderr,"%f+%fi ", testData [ i*alignMat + j*LOAD_SIZE + k].x, \
							testData [ i*alignMat + j*LOAD_SIZE + k].y );
				}
				fprintf(stderr,"\n");
			}
			fprintf(stderr,"\n");
		}
		free(testData);
#endif
}



void setupM_Matrix ( float * theta, float * phi, float2 * u, float2 * v, int Q, float2 * buf )
{
	//size of buf is 2*MATRIX_SIDE*MATRIXSIDE
	for (int i=0; i< 2*MATRIX_SIDE; i++)
	{
		float2 zplus	= divide( u[i], v[i] );
		//zplus=u/v for ith particle
		float2 zminus 	= divide( v[i], u[i] );
		//zminus=v/u for ith particle

		buf[i* MATRIX_SIDE+Q+1].x = powf( cos( theta[i] ) * sin( theta[i] ), Q );
		buf[i* MATRIX_SIDE+Q+1].y = 0.0f;
		//u^(Q+m) X v^(Q-m) where m=0; /*********Reminder for myself: this excludes the case of Q being a half integer.***********/
		for (int j=1; j<(Q+2); j++)
		{
			buf[ i* MATRIX_SIDE+Q+1 + j] = multiply (buf[ i*MATRIX_SIDE +(Q+1)+ j-1 ], zplus);
			//for i=0, this fills in the entries from (Q+1+1) to (Q+1+Q+1). They correspond to angular momenta 1 to Q+1
			//for i=2*MATRIX_SIDE-1 this fills in the entries from (2*MATRIX_SIDE*MATRIX_SIDE-MATRIX_SIDE+Q+1+1) to (2*MATRIX_SIDE*MATRIX_SIDE-MATRIX_SIDE+Q+1+Q+1)
			//since 2Q+2=MATRIX_SIDE-1, this corresponds to (2*MATRIX_SIDE*MATRIX_SIDE-MATRIX_SIDE+MATRIX_SIDE-1)=(2*MATRIX_SIDE*MATRIX_SIDE-1) which is infact the last element of the vector
			buf[ i* MATRIX_SIDE+Q+1 - j] = multiply (buf[ i*MATRIX_SIDE +(Q+1) - j+1 ], zminus);
			//for i=0, this fills in the entries from (Q+1-1) to (Q+1-Q-1). They correspond to angular momenta -1 to -(Q+1)
			//for the current test setting : MATRIX_SIDE = 11 and Q=(11-3)/2=4. 2Q+3=MATRIX_SIDE

		}
	}
}

__device__ float2 calculateElement(int rowIndex, int colIndex, int * perm){

	// 0-based indexing ergo the -1 everywhere
	// assemble second last (N-2) and last (N-1) columns &add to rest of M

	int i= perm[rowIndex] - 1;
	int Q=(MATRIX_SIDE-3)/2;
	if(colIndex<(MATRIX_SIDE-2))
	{
		return (deviceM[ i * MATRIX_SIDE + colIndex ]);
	}

	float2 ret1,ret2,tmp,ui,vi,uj,vj;

	ui = deviceU[ i ];
	vi = deviceV[ i ];

	ret1.x = 0.0f; ret1.y = 0.0f;
	ret2.x = 0.0f; ret2.y = 0.0f;

	// do the sum for N-2 column
	for (int j=0; j<rowIndex; j++){

		uj = deviceU[ perm[j] -1 ];
		vj = deviceV[ perm[j] -1 ];

		// denominator
		tmp =  subtract ( multiply ( ui, vj), multiply ( uj, vi ));
		//tmp=ui*vj-vi*uj

		// sum j<i
		ret1 = add ( ret1, divide ( add ( multiply ( multiply ( dk1, ui ), vj), \
			multiply ( multiply ( dk2, uj ), vi )), tmp ));

		//ret1 + = (dk1*ui*vj+dk2*uj*vi)/(ui*vj-uj*vi);
		//test dk1=1 dk2=0.5;
		//note to myself : above step can be simplified.
	}

	for (int j=rowIndex+1; j<MATRIX_SIDE; j++){

		uj = deviceU[ perm[j] -1 ];
		vj = deviceV[ perm[j] -1 ];

		// denominator
		tmp =  subtract ( multiply ( ui, vj), multiply ( uj, vi ));

		// sum j>i
		ret1 = add ( ret1, divide ( add ( multiply ( multiply ( dk1, ui ), vj), \
			multiply ( multiply ( dk2, uj), vi )), tmp ));
		//note to myself : above step can be simplified.
	}

	// keep only N-2 (second last) column
//	ret1 = multiply ( ret1, threadMaskA[ colIndex ] );

	// multiply by factor from lookup
	ret1 = multiply ( ret1, deviceM[ i * MATRIX_SIDE + (Q+1)+dmu ]);

	// repeat for N-1
	for (int j=0; j<rowIndex; j++){

		uj = deviceU[ perm[j] -1 ];
		vj = deviceV[ perm[j] -1 ];

		// denominator
		tmp =  subtract ( multiply ( ui, vj), multiply ( uj, vi ));

		// sum j < i
		ret2 = add ( ret2, divide ( add ( multiply ( multiply ( dr1, ui ), vj), \
			multiply ( multiply ( dr2, uj ), vi )), tmp ));
	}


	for (int j=rowIndex+1; j<MATRIX_SIDE; j++){

		uj = deviceU[ perm[j] -1 ];
		vj = deviceV[ perm[j] -1 ];


		// denominator
		tmp =  subtract ( multiply ( ui, vj), multiply ( uj, vi ));

		// sum j > i
		ret2 = add ( ret2, divide ( add ( multiply ( multiply ( dr1, ui ), vj), \
			multiply ( multiply ( dr2, uj ), vi ) ), tmp ));
	}

	// keep only N-1 (last) column
//	ret2 = multiply ( ret2, threadMaskB[ colIndex ]);

	// multiply by factor from lookup
	ret2 = multiply ( ret2, deviceM[ i * MATRIX_SIDE + (Q+1)+ dnu ]);

	// lookup elements from M and assemble complete matrix
	return add ( ret1,ret2);

}

__global__ void makeMatrices(int offset, float2 * devMatrices, short * devPermutations){


	__shared__ int scratch[MATRIX_SIDE];

	int rowIndex = threadIdx.x / MATRIX_SIDE;
	int colIndex = threadIdx.x % MATRIX_SIDE;

	// load permutation for this block
	if (threadIdx.x < MATRIX_SIDE){

		scratch [ threadIdx.x ] = (int) devPermutations [( offset + blockIdx.x )* LOAD_SIZE + threadIdx.x ];
	}

	__syncthreads();



#ifdef TEST_MAKE_MATRICES
	if ((blockIdx.x ==0) && (threadIdx.x==0)){

		printf("perms on device\n");
		for (int i=0; i<MATRIX_SIDE*LOAD_SIZE*2; i++) printf("%f ",devPermutations[i]);

		printf("\n");
		printf("lookup tables on device :\n");
		printf("threadMaskA.x\n");
		for (int i=0; i<MATRIX_SIDE; i++) printf("%f ",threadMaskA[i].x);
		printf("\n");
		printf("threadMaskB.x\n");
		for (int i=0; i<MATRIX_SIDE; i++) printf("%f ",threadMaskB[i].x);
		printf("\n");
		printf("deviceA\n");
		for (int i=0; i<2*MATRIX_SIDE; i++) printf("%f ",deviceA[i]);
		printf("\n");
		printf("deviceU\n");
		for (int i=0; i<2*MATRIX_SIDE; i++) printf("%f+%fi ",deviceU[i].x,deviceU[i].y);
		printf("\n");
		printf("deviceV\n");
		for (int i=0; i<2*MATRIX_SIDE; i++) printf("%f+%fi ",deviceV[i].x,deviceV[i].y);
		printf("\n");
		printf("matrixM\n");
		for (int i=0; i<2*MATRIX_SIDE; i++){
			for (int j=0; j<MATRIX_SIDE; j++)
				printf("%f+%fi ",deviceM[i*MATRIX_SIDE + j].x,deviceM[i*MATRIX_SIDE + j].y);

			printf("\n");
		}
	}

#endif
	// clear mem
	//if (threadIdx.x < LOAD_SIZE * MATRIX_SIDE){
	//	devMatrices[threadIdx.x + (blockIdx.x + offset) * LOAD_SIZE*MATRIX_SIDE].x = 0.0f;
	//	devMatrices[threadIdx.x + (blockIdx.x + offset) * LOAD_SIZE*MATRIX_SIDE].y = 0.0f;
	//}

	__syncthreads();

	// calculate and write out matrix
	if (threadIdx.x < MATRIX_SIDE*MATRIX_SIDE)
		devMatrices[(blockIdx.x + offset)*LOAD_SIZE*MATRIX_SIDE + rowIndex * LOAD_SIZE + colIndex] \
			= calculateElement(rowIndex, colIndex,scratch);
}


}

#endif

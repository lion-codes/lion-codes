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




#include "main.h"

/* Compute the eigenspectrum for a hermitian matrix using lanczos & tridiagonal qr algorithm

Requirements :
----------------------------------------------------------------------------------------
OpenMPI built with cuda support
NVIDIA Driver >= v270.41.19
RH Linux Kernel  >= 2.6.15

On lion-GA at PSU:
-----------------------------------------------------------------------------------------
module load cuda/5.0
module load mkl
export LD_LIBRARY_PATH=/usr/global/openmpi/022513/gnu/lib/:$LD_LIBRARY_PATH
export PATH=/usr/global/openmpi/022513/gnu/bin/:$PATH


References : 
----------------------------------------------------------------------------------------
Matrix Computations, Golub & Van Loan
Peter Arbenz (ETH) lecture notes
Mellanox/GPUdirect http://www.mellanox.com/pdf/whitepapers/TB_GPU_Direct.pdf
OpenMPI & CUDA build http://www.open-mpi.org/faq/?category=building#build-cuda
OpenMPI & CUDA support http://www.open-mpi.org/faq/?category=running#mpi-cuda-support


*/


#ifdef _USE_GPU

// cuda errors
void check_last_cuda_error(cudaError_t cerror, char * msg, char * hostname,int line){


	if (cerror != cudaSuccess){
		fprintf(stderr,"%s %s with code : %s in file %s around line %d \n",msg,hostname,cudaGetErrorString( cerror),__FILE__,line);
		MPI_Finalize();
		exit(1);
	}


}

// cublas errors
void check_last_cublas_error(cublasStatus_t status, char * msg, char * hostname,int line){

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "%s %s with code : %s in file %s around line %d \n",msg,hostname,cudaGetErrorString( status),__FILE__,line);
		MPI_Finalize();
		exit(1);
	}


}




#endif


void lanczos(complex double * A, 	// chunk of A
		int matSize, 			// full size of A
		int myRows,			// rows of A for this process
		int myOffset,			// where to begin			
		int subSize,			// the subspace size
		int commSize,			// MPI size
		int commRank){			// MPI rank


	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


#ifdef _USE_GPU
	// check the device
	char hostname[256];
	gethostname(hostname,255);

	struct cudaDeviceProp p;
	cudaGetDeviceProperties(&p,0);
	int support = p.canMapHostMemory;

	if(support == 0){
		fprintf(stderr,"%s does not support mapping host memory\n",hostname);
		MPI_Finalize();
		exit(1);
	}

#endif

	// malloc vectors for use in lanczos
	complex double * alpha	= (complex double*) malloc(sizeof(complex double) * subSize);
	complex double * evecs;

#ifdef _CALC_EVECS
	evecs = (complex double*) malloc(sizeof(complex double) * subSize * subSize);
#endif

	complex double * beta	= (complex double*) malloc(sizeof(complex double) * (subSize-1));
	complex double * r ;

#ifndef _USE_GPU
	r = (complex double*) malloc(sizeof(complex double) * matSize);
#endif

	complex double * scratch= (complex double*) malloc(sizeof(complex double) * matSize);
	complex double * Q	= (complex double*) malloc(sizeof(complex double) * matSize * subSize);

	for (int i=0; i<matSize*subSize; i++)
		Q[i] = 0.0+0.0*_Complex_I;


	// an initial q-vector in first column of Q
	for (int i=0; i<matSize; i++)
		Q[i] = (1.0+1.0*_Complex_I) / sqrt(2.0f* (double) matSize);



#ifdef _USE_GPU

	cudaError_t cerror;
	cublasStatus_t status = cublasInit();
#ifdef _DEBUG_LANCZOS
	check_last_cublas_error(status,"CUBLAS initialization error on host",hostname,__LINE__);

#endif
	cuDoubleComplex d_ortho;
	cuDoubleComplex * d_r;
	cuDoubleComplex * d_A;
	cuDoubleComplex * d_Q;
	cuDoubleComplex * d_beta;
	cuDoubleComplex * d_alpha;
	cuDoubleComplex * d_output;

	// zero copy memory for vector r, for use with MPI
	cerror = cudaHostAlloc((void**) &r,sizeof(cuDoubleComplex)*matSize,cudaHostAllocMapped);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaHostAlloc failed for r on host",hostname,__LINE__);

#endif
	cerror = cudaHostGetDevicePointer(&d_r,r,0);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaHostGetDevicePointer failed for d_r on host",hostname,__LINE__);

#endif
	// regular mallocs for everyone else
	cerror = cudaMalloc((void**) &d_ortho, sizeof(cuDoubleComplex));
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_ortho on host",hostname,__LINE__);

#endif
	cerror = cudaMalloc((void**) &d_alpha, sizeof(cuDoubleComplex) * subSize);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_alpha on host",hostname,__LINE__);

#endif
	cerror = cudaMalloc((void**) &d_beta, sizeof(cuDoubleComplex) * (subSize-1));
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_beta on host",hostname,__LINE__);

#endif
	cerror = cudaMalloc((void**) &d_Q, sizeof(cuDoubleComplex) * subSize*matSize);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_Q on host",hostname,__LINE__);

#endif
	cerror = cudaMalloc((void**) &d_A, sizeof(cuDoubleComplex) * myRows * matSize);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_A on host",hostname,__LINE__);

#endif
	cerror = cudaMalloc((void**) &d_output, sizeof(cuDoubleComplex) * matSize);
#ifdef _DEBUG_LANCZOS
	check_last_cuda_error(cerror,"cudaMalloc failed for d_output on host",hostname,__LINE__);

#endif
	// gpu running configuration
	cublasHandle_t handle;
	cublasCreate(&handle);

	dim3 threads,blocks;
	threads.x 	= _LAN_THREADS;
	blocks.x 	= matSize / threads.x +1;

	threads.y	= 1;
	threads.z	= 1;
	blocks.y	= 1;
	blocks.z	= 1;

#endif

	// multiplicative factors in gemv
	complex double mula 	= 1.0+0.0*_Complex_I;
	complex double mulb 	= 0.0+0.0*_Complex_I;
	complex double mulc 	= -1.0+0.0*_Complex_I;

	// args for gemv
	char type = 'N';
	int m=myRows,n=matSize,info;
	int inc=1,dim=matSize;


	// init vectors
	zgemv_(&type,&myRows,&n,&mula,A,&myRows,Q,&inc,&mulb,r,&inc);


	// need to gather into r
	int success = MPI_Allgather((void*) r, myRows, MPI_LONG_DOUBLE, \
			(void*) r, myRows, MPI_LONG_DOUBLE,MPI_COMM_WORLD);



#ifdef _DEBUG_LANCZOS
	if (success != MPI_SUCCESS) {

		char error_string[256];
		int length_of_error_string;

		MPI_Error_string(success, error_string, &length_of_error_string);
		fprintf(stderr,"MPI_Allgather failed in file %s around line %d with code : %s\n",__FILE__,__LINE__,error_string);
		MPI_Finalize();
		exit(1);
	}

#endif
	for (int i=0; i<subSize; i++) alpha[i] 	= 0.0f;
	for (int i=0; i<subSize-1; i++) beta[i] = 0.0f;

	for (int i=0; i<n; i++) alpha[0] 	+= r[i] * conj(Q[i]);
	for (int i=0; i<n; i++) r[i] 		-= alpha[0] * Q[i];
	for (int i=0; i<n; i++) beta[0]		+= conj(r[i]) * r[i];	
	beta[0] = sqrt(beta[0]);

	//test subsequent lanczos vectors
	complex double ortho;

#ifdef _USE_GPU

	// send to device
	status =cublasSetVector(subSize,sizeof(cuDoubleComplex),alpha,1.0,d_alpha,1.0);
#ifdef _DEBUG_LANCZOS
	check_last_cublas_error(status,"cublasSetVector failed for d_alpha on host",hostname,__LINE__);

#endif
	status =cublasSetVector(subSize-1,sizeof(cuDoubleComplex),beta,1.0,d_beta,1.0);
#ifdef _DEBUG_LANCZOS
	check_last_cublas_error(status,"cublasSetVector failed for d_beta on host",hostname,__LINE__);

#endif
	status = cublasSetMatrix(myRows,matSize,sizeof(cuDoubleComplex),A,myRows,d_A,myRows);
#ifdef _DEBUG_LANCZOS
	check_last_cublas_error(status,"cublasSetMatrix failed for d_A on host",hostname,__LINE__);

#endif
	status = cublasSetMatrix(matSize,subSize,sizeof(cuDoubleComplex),Q,matSize,d_Q,matSize);
#ifdef _DEBUG_LANCZOS
	check_last_cublas_error(status,"cublasSetMatrix failed for d_Q on host",hostname,__LINE__);

#endif
#endif


	// main lanczos loops
	for (int i=1; i<subSize; i++){



		MPI_Barrier(MPI_COMM_WORLD);
		ortho = 0.0+0.0*_Complex_I;

#ifndef _USE_GPU


		// new column to Q, updated q
		for (int j=0; j<n; j++) Q[i*n+j] = r[j] / beta[i-1];

		// update r 
		zgemv_(&type,&myRows,&n,&mula,A,&myRows,&Q[i*n],&inc,&mulb,r,&inc);

		// need to gather into r
		int success = MPI_Allgather((void*) r, myRows, MPI_LONG_DOUBLE, \
				(void*) r, myRows, MPI_LONG_DOUBLE,MPI_COMM_WORLD);


#ifdef _DEBUG_LANCZOS
		if (success != MPI_SUCCESS) {

			char error_string[256];
			int length_of_error_string;

			MPI_Error_string(success, error_string, &length_of_error_string);
			fprintf(stderr,"MPI_Allgather failed in file %s around line %d with code : %s\n",__FILE__,__LINE__,error_string);
			MPI_Finalize();
			exit(1);
		}

#endif
		// another r update
		for (int j=0; j<n; j++) r[j] 	-= beta[i] * Q[(i-1)*n+j];

		// update alpha
		for (int j=0; j<n; j++) alpha[i]+= r[j] * conj(Q[i*n+j]);

		// r update
		for (int j=0; j<n; j++) r[j] 	-= alpha[i] * Q[i*n+j];

		// weak orthogonality test
		//
		for (int j=0; j<n; j++)	ortho 	+= conj(Q[(i-1)*n+j]) * Q[i*n+j];




		// re-orthogonalize
		// r -= Q(Q^T * r)
		//if (fabs(ortho) > _EVECS_NORM){
		if (0){

			char typet = 'C';
			zgemv_(&typet,&n,&subSize,&mula,Q,&dim,r,&inc,&mulb,scratch,&inc);
			zgemv_(&type,&n,&subSize,&mulc,Q,&dim,scratch,&inc,&mula,r,&inc);


		}

		// update beta
		if (i<subSize-1){
			for (int j=0; j<n; j++) beta[i]	+= conj(r[j]) * r[j];	
			beta[i] = sqrt(beta[i]);
		}

#else
		cerror = lanczos_first_update(blocks, threads, d_r, d_Q, d_beta, matSize, i);

#ifdef _DEBUG_LANCZOS
		check_last_cuda_error(cerror,"lanczos_first_update failed on host",hostname,__LINE__);
#endif

		cublasGetError();


		cublasZgemv(handle,CUBLAS_OP_N,myRows,n,&mula,d_A,myRows,&d_Q[i*m],1,&mulb,&d_r[myOffset],1);

		status = cublasGetError();
#ifdef _DEBUG_LANCZOS
		check_last_cublas_error(status,"cublasZgemv failed on host",hostname,__LINE__);
#endif

#if 0
		{
			int i = 0;
			char hostname[256];
			gethostname(hostname, sizeof(hostname));
			printf("PID %d on %s ready for attach\n", getpid(), hostname);
			fflush(stdout);
			while (0 == i)
				sleep(5);
		}

#endif
		// need to gather into r
		int success = MPI_Allgather((void*) d_r, myRows, MPI_LONG_DOUBLE, (void*) d_r, myRows, MPI_LONG_DOUBLE,MPI_COMM_WORLD);
#ifdef _DEBUG_LANCZOS
		if (success != MPI_SUCCESS) {

			char error_string[256];
			int length_of_error_string;

			MPI_Error_string(success, error_string, &length_of_error_string);
			fprintf(stderr,"gpu MPI_Allgather failed in file %s around line %d with code %s\n",__FILE__,__LINE__,error_string);
			MPI_Finalize();
			exit(1);
		}

#endif

		cerror = lanczos_second_update(blocks, threads, d_r, d_Q, d_beta, matSize, i);
#ifdef _DEBUG_LANCZOS
		check_last_cuda_error(cerror,"lanczos_second_update failed on host",hostname,__LINE__);
#endif
		cerror = vector_dot(d_Q,d_r,d_output,&d_alpha[i],1,matSize,i*n,0,0);
		check_last_cuda_error(cerror,"vector_dot failed on host",hostname,__LINE__);
#ifdef _DEBUG_LANCZOS
#endif

		cerror = lanczos_third_update(blocks, threads, d_r, d_Q, d_alpha, matSize, i);
#ifdef _DEBUG_LANCZOS
		check_last_cuda_error(cerror,"lanczos_third_update failed on host",hostname,__LINE__);
#endif

		if (i<subSize-1){
		cerror = vector_dot(d_r,d_r,d_output,&d_beta[i],1,matSize,0,0,1);
		}

#ifdef _DEBUG_LANCZOS
		check_last_cuda_error(cerror,"vector_dot failed on host",hostname,__LINE__);
#endif


		// crude orthogonality test
		//
		if (0){
			cerror = vector_dot(d_Q,d_Q,d_output,&d_ortho,1,matSize,(i-1)*m,i*m,0);
#ifdef _DEBUG_LANCZOS
			check_last_cuda_error(cerror,"vector_dot failed on host",hostname,__LINE__);
#endif

			cudaMemcpy(&ortho,&d_ortho,sizeof(complex double), cudaMemcpyDeviceToHost);

			if (fabs(ortho) > _EVECS_NORM){


				cublasGetError();

				cublasZgemv(handle,CUBLAS_OP_T,n,subSize,&mula,d_Q,dim,d_r,1,&mulb,d_output,1);
				cublasZgemv(handle,CUBLAS_OP_N,n,subSize,&mula,d_Q,dim,d_output,1,&mulb,d_output,1);

				status = cublasGetError();
#ifdef _DEBUG_LANCZOS
				check_last_cublas_error(status,"cublasZgemv failed on host",hostname,__LINE__);
#endif

				cerror = lanczos_fourth_update(blocks, threads, d_r, d_output, matSize);
#ifdef _DEBUG_LANCZOS
				check_last_cuda_error(cerror,"lanczos_fourth_update failed on host",hostname,__LINE__);
#endif
			}


		}
#endif
		}

#ifdef _USE_GPU

		if (commRank==0){

			cerror = cudaMemcpy(alpha,d_alpha,sizeof(cuDoubleComplex) * subSize, cudaMemcpyDeviceToHost);
#ifdef _DEBUG_LANCZOS
			check_last_cuda_error(cerror,"cudaMemcpy of d_alpha to host",hostname,__LINE__);
#endif
			cerror = cudaMemcpy(beta,d_beta,sizeof(cuDoubleComplex) * (subSize-1), cudaMemcpyDeviceToHost);
#ifdef _DEBUG_LANCZOS
			check_last_cuda_error(cerror,"cudaMemcpy of d_beta to host",hostname,__LINE__);
#endif
			cerror = cudaMemcpy(Q,d_Q,sizeof(cuDoubleComplex) * subSize*matSize, cudaMemcpyDeviceToHost);
#ifdef _DEBUG_LANCZOS
			check_last_cuda_error(cerror,"cudaMemcpy of d_Q to host",hostname,__LINE__);
#endif

		}
		cudaFree(d_alpha);
		cudaFree(d_output);
		cudaFree(d_beta);
		cudaFree(d_Q);
		cudaFreeHost(d_r);
		cudaFree(d_A);

#endif

		if (commRank==0){
#ifdef _DEBUG_LANCZOS
			printf("alpha & beta :\n");
			for (int i=0; i<subSize; i++)
				printf("%f+%fi ",creal(alpha[i]),cimag(alpha[i]));
			printf("\n");
			for (int i=0; i<subSize-1; i++)
				printf("%f+%fi ",creal(beta[i]),cimag(beta[i]));
			printf("\n");

#endif

			// calculate spectrum of (now) tridiagonal matrix
			//wilk_qr(alpha, beta, evecs,subSize);


#ifdef _DEBUG_LANCZOS
			printf("evals :\n");

			for (int i=0; i<subSize; i++)
				printf("%f+%fi ",creal(alpha[i]),cimag(alpha[i]));
			printf("\n");
#endif

#ifdef _CALC_EVECS

			// use the lanczos evectors to build evecs for A




#endif

		}


		free(alpha); 	
#ifdef _CALC_EVECS
		free(evecs);
#endif

		//free(beta);
#ifndef _USE_GPU
		free(r);
#endif
		//free(Q);
		//free(scratch);
	}

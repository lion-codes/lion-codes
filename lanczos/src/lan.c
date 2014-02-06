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
// Compute the eigenspectrum for a hermitian matrix using lanczos algorithm &tridiagonal methods
#ifdef _USE_GPU

// cuda errors
void check_last_cuda_error(cudaError_t cerror, char * msg, char * hostname,int line){


	if (cerror != cudaSuccess){
		fprintf(stderr,"%s %s with code : %s in file %s around line %d \n",\
				msg,hostname,cudaGetErrorString( cerror),__FILE__,line);
		MPI_Finalize();
		exit(1);
	}


}

// cublas errors
void check_last_cublas_error(cublasStatus_t status, char * msg, char * hostname,int line){

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "%s %s with code : %s in file %s around line %d \n",\
				msg,hostname,cudaGetErrorString( status),__FILE__,line);
		MPI_Finalize();
		exit(1);
	}


}




#endif


void lanczos(complex double * A, 	// chunk of A
		complex double * evecs, //the eigenvectors
		double * evals,		//evals, real
		int n, 			// full size of A
		int m,			// rows of A for this process
		int myOffset,			// where to begin			
		int subSize,			// the subspace size
		int commSize,			// MPI size
		int commRank){			// MPI rank


	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	// args for gemv
	char type = 'N';
	int info,inc=1,dim=n;



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
	complex double * beta	= (complex double*) malloc(sizeof(complex double) * (subSize-1));
	complex double * r ;

	r = (complex double*) malloc(sizeof(complex double) * n);

	complex double * scratch= (complex double*) malloc(sizeof(complex double) * n);
	complex double * Q	= (complex double*) malloc(sizeof(complex double) * n * subSize);

	for (int i=0; i<m*n; i++)
		Q[i] = 0.0+0.0*_Complex_I;


	// an initial q-vector in first column of Q
	for (int i=0; i<n; i++)
		Q[i] = (1.0+1.0*_Complex_I) / sqrt(2.0f* (double) n);


	//dump_mat("Q",Q);

#ifdef _USE_GPU

	cudaError_t cerror;
	cublasStatus_t status = cublasInit();
	check_cu_error("CUBLAS initialization error on host");

	cuDoubleComplex d_ortho;
	cuDoubleComplex * d_r;
	cuDoubleComplex * d_A;
	cuDoubleComplex * d_Q;
	cuDoubleComplex * d_beta;
	cuDoubleComplex * d_alpha;
	cuDoubleComplex * d_output;

	// zero copy memory for vector r, for use with MPI
	cerror = cudaHostAlloc((void**) &r,sizeof(cuDoubleComplex)*n,cudaHostAllocMapped);
	check_cu_error("cudaHostAlloc failed for r on host");
	cerror = cudaHostGetDevicePointer(&d_r,r,0);
	check_cu_error("cudaHostGetDevicePointer failed for d_r on host");
	// regular mallocs for everyone else
	cerror = cudaMalloc((void**) &d_ortho, sizeof(cuDoubleComplex));
	check_cu_error("cudaMalloc failed for d_ortho on host");
	cerror = cudaMalloc((void**) &d_alpha, sizeof(cuDoubleComplex) * subSize);
	check_cu_error("cudaMalloc failed for d_alpha on host");
	cerror = cudaMalloc((void**) &d_beta, sizeof(cuDoubleComplex) * (subSize-1));
	check_cu_error("cudaMalloc failed for d_beta on host");

	cerror = cudaMalloc((void**) &d_Q, sizeof(cuDoubleComplex) * subSize*n);
	check_cu_error("cudaMalloc failed for d_Q on host");
	cerror = cudaMalloc((void**) &d_A, sizeof(cuDoubleComplex) * m * n);
	check_cu_error("cudaMalloc failed for d_A on host");
	cerror = cudaMalloc((void**) &d_output, sizeof(cuDoubleComplex) * n);
	check_cu_error("cudaMalloc failed for d_output on host");
	// gpu running configuration
	cublasHandle_t handle;
	cublasCreate(&handle);

	dim3 threads,blocks;
	threads.x 	= _LAN_THREADS;
	blocks.x 	= n / threads.x +1;

	threads.y,threads.z,blocks.y,blocks.z	= 1;

		printf("blocks: %i threads: %i\n",blocks.x,threads.x);
#endif

	// multiplicative factors in gemv
	complex double mula 	= 1.0+0.0*_Complex_I;
	complex double mulb 	= 0.0+0.0*_Complex_I;
	complex double mulc 	= -1.0+0.0*_Complex_I;

	// args for gemv
	//char type = 'N';
	//int m=m,n=n,info;
	//int inc=1,dim=n;


	// init vectors
	zgemv_(&type,&m,&n,&mula,A,&m,Q,&inc,&mulb,&r[myOffset],&inc);


	// need to gather into r
	int success = MPI_Allgather((void*) &r[myOffset], m, MPI_LONG_DOUBLE, \
			(void*) r, m, MPI_LONG_DOUBLE,MPI_COMM_WORLD);

	//dump_vec(commRank,"r",r);


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


	//dump_vec(commRank,"alpha",alpha);

	//test subsequent lanczos vectors
	double ortho;

#ifdef _USE_GPU

	// send to device
	status =cublasSetVector(subSize,sizeof(cuDoubleComplex),alpha,1.0,d_alpha,1.0);
	check_last_cublas_error(status,"cublasSetVector failed for d_alpha on host",hostname,__LINE__);
	status =cublasSetVector(subSize-1,sizeof(cuDoubleComplex),beta,1.0,d_beta,1.0);
	check_cb_error("cublasSetVector failed for d_beta on host");
	status = cublasSetMatrix(m,n,sizeof(cuDoubleComplex),A,m,d_A,m);
	check_cb_error("cublasSetMatrix failed for d_A on host");
	status = cublasSetMatrix(n,subSize,sizeof(cuDoubleComplex),Q,n,d_Q,n);
	check_cb_error("cublasSetMatrix failed for d_Q on host");
#endif


#ifdef _GATHER_SCALAR
	//reduction not currently supported for cuda
	complex double * alpha_temp = (complex double * ) malloc (sizeof(complex double) * commSize);
	complex double * beta_temp = (complex double * ) malloc (sizeof(complex double) * commSize);

#endif
	// main lanczos loops
	for (int i=1; i<subSize; i++){

		MPI_Barrier(MPI_COMM_WORLD);
		ortho = 0.0;

#ifndef _USE_GPU


		// new column to Q, updated q
		for (int j=0; j<n; j++) Q[i*n+j] = r[j] / beta[i-1];

		// update r 
		zgemv_(&type,&m,&n,&mula,A,&m,&Q[i*n],&inc,&mulb,&r[myOffset],&inc);

#ifndef _GATHER_SCALAR
		// need to gather into r
		int success = MPI_Allgather((void*) &r[myOffset], m, MPI_LONG_DOUBLE, \
				(void*) r, m, MPI_LONG_DOUBLE,MPI_COMM_WORLD);


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

#endif
		// another r update
		for (int j=0; j<n; j++) r[j] 	-= beta[i] * Q[(i-1)*n+j];

#ifndef _GATHER_SCALAR
		// update alpha
		for (int j=0; j<n; j++) alpha[i]+= r[j] * conj(Q[i*n+j]);

#else
		alpha_temp[commRank]=0.0+0.0*I;
		for (int j=0; j<m; j++) alpha_temp[commRank] +=r[j+myOffset] * conj(Q[i*n+j+myOffset]);
		// need to gather into r
		int success = MPI_Allgather((void*) &alpha_temp[commRank], 1, MPI_LONG_DOUBLE, \
				(void*) alpha_temp, commSize-1, MPI_LONG_DOUBLE,MPI_COMM_WORLD);

		for (int j=0; j<commSize; j++) alpha[i]+=alpha_temp[j];


#endif

		// r update
		for (int j=0; j<n; j++) r[j] 	-= alpha[i] * Q[i*n+j];

		// weak orthogonality test
		for (int j=0; j<n; j++)	ortho 	+= fabs(conj(Q[j]) * Q[i*n+j]);



		// re-orthogonalize
		// r -= Q(Q^T * r)
		if (fabs(ortho) > _EVECS_NORM){

#ifdef _GATHER_SCALAR
			// need to gather into r
			int success = MPI_Allgather((void*) &r[myOffset], m, MPI_LONG_DOUBLE, \
					(void*) r, m, MPI_LONG_DOUBLE,MPI_COMM_WORLD);


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

#endif

			//if (1){

			char typet = 'C';
			zgemv_(&typet,&n,&subSize,&mula,Q,&dim,r,&inc,&mulb,scratch,&inc);
			zgemv_(&type,&n,&subSize,&mulc,Q,&dim,scratch,&inc,&mula,r,&inc);


		}

		// update beta
		if (i<subSize-1){

#ifndef _GATHER_SCALAR

			for (int j=0; j<n; j++) beta[i]	+= conj(r[j]) * r[j];	

#else

			beta_temp[commRank]=0.0+0.0*I;
			for (int j=0; j<m; j++) beta_temp[commRank] +=conj(r[j+myOffset]) * r[j+myOffset];
			int success = MPI_Allgather((void*) &beta_temp[commRank], 1, MPI_LONG_DOUBLE, \
					(void*) beta_temp, commSize-1, MPI_LONG_DOUBLE,MPI_COMM_WORLD);

			for (int j=0; j<commSize; j++) beta[i]+=beta_temp[j];


#endif
			beta[i] = sqrt(beta[i]);
		}

#else

		cerror = lanczos_first_update(blocks, threads, d_r, d_Q, d_beta, n, i);
		check_cu_error("lanczos_first_update failed on host");

		cublasGetError();


		cublasZgemv(handle,CUBLAS_OP_N,m,n,&mula,d_A,m,&d_Q[i*m],1,&mulb,&d_r[myOffset],1);

		status = cublasGetError();
		check_cb_error("cublasZgemv failed on host");

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
		int success = MPI_Allgather((void*) &d_r[myOffset], m, MPI_LONG_DOUBLE, (void*) d_r, m, MPI_LONG_DOUBLE,MPI_COMM_WORLD);



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

		cerror = lanczos_second_update(blocks, threads, d_r, d_Q, d_beta, n, i);
		check_cu_error("lanczos_second_update failed on host");
		cerror = vector_dot(d_Q,d_r,d_output,&d_alpha[i],1,n,i*n,0,0);
		check_cu_error("vector_dot failed on host");

		cerror = lanczos_third_update(blocks, threads, d_r, d_Q, d_alpha, n, i);
		check_cu_error("lanczos_third_update failed on host");

		if (i<subSize-1){
			cerror = vector_dot(d_r,d_r,d_output,&d_beta[i],1,n,0,0,1);
		}

		check_cu_error("vector_dot failed on host");


		// crude orthogonality test
		//
		if (1){
			cerror = vector_dot(d_Q,d_Q,d_output,&d_ortho,1,n,(i-1)*m,i*m,0);
			check_cu_error("vector_dot failed on host");

			cudaMemcpy(&ortho,&d_ortho,sizeof(complex double), cudaMemcpyDeviceToHost);

			if (fabs(ortho) > _EVECS_NORM){


				cublasGetError();

				cublasZgemv(handle,CUBLAS_OP_T,n,subSize,&mula,d_Q,dim,d_r,1,&mulb,d_output,1);
				cublasZgemv(handle,CUBLAS_OP_N,n,subSize,&mula,d_Q,dim,d_output,1,&mulb,d_output,1);

				status = cublasGetError();
				check_cb_error("cublasZgemv failed on host");

				cerror = lanczos_fourth_update(blocks, threads, d_r, d_output, n);
				check_cu_error("lanczos_fourth_update failed on host");
			}


		}
#endif
		}

#ifdef _USE_GPU

		if (commRank==0){

			cerror = cudaMemcpy(alpha,d_alpha,sizeof(cuDoubleComplex) * subSize, cudaMemcpyDeviceToHost);
			check_cu_error("cudaMemcpy of d_alpha to host");
			cerror = cudaMemcpy(beta,d_beta,sizeof(cuDoubleComplex) * (subSize-1), cudaMemcpyDeviceToHost);
			check_cu_error("cudaMemcpy of d_beta to host");
			cerror = cudaMemcpy(Q,d_Q,sizeof(cuDoubleComplex) * subSize*n, cudaMemcpyDeviceToHost);
			check_cu_error("cudaMemcpy of d_Q to host");

		}
		cudaFree(d_alpha);
		cudaFree(d_output);
		cudaFree(d_beta);
		cudaFree(d_Q);
		cudaFreeHost(d_r);
		cudaFree(d_A);

#endif

#ifdef _DEBUG_LANCZOS
if (commRank==0){

		printf("alpha & beta :\n");
		for (int i=0; i<subSize; i++)
			printf("%f+%fi ",creal(alpha[i]),cimag(alpha[i]));
		printf("\n");
		for (int i=0; i<subSize-1; i++)
			printf("%f+%fi ",creal(beta[i]),cimag(beta[i]));
		printf("\n");
}
#endif
		// calculate spectrum of (now) tridiagonal matrix

		double * alp = (double*) malloc(sizeof(double) * subSize);
		double * bet = (double*) malloc(sizeof(double) * (subSize-1));

		for (int i=0; i<subSize; i++) alp[i] = creal(alpha[i]);
		for (int i=0; i<(subSize-1); i++) bet[i] = creal(beta[i]);

#ifdef _CALC_EVECS

		complex double * evecs_lan = (complex double*) malloc(sizeof(complex double) * subSize * subSize);


		type = 'I';

		zsteqr_(&type,&subSize,alp,bet,evecs_lan,&subSize,(double*) evecs,&info);

		type = 'N';

		for (int i=0; i<subSize; i++)
			zgemv_(&type,&n,&subSize,&mula,Q,&n,&evecs_lan[i*subSize],&inc,&mulb,&evecs[i*n],&inc);

		free(evecs_lan);
#else

		dsterf_(&subSize,alp,bet,&info);
		free(bet);

#endif

		for (int i=0; i<subSize; i++) evals[i] = alp[i];

#ifdef _DEBUG_LANCZOS
		
if (commRank==0){
		printf("evals :\n");

		for (int i=0; i<subSize; i++)
			printf("%f ",evals[i]);
		printf("\n");

}
#endif


		free(alp); 
		free(alpha); 	
		free(beta);
		free(r);
		free(Q);
	}

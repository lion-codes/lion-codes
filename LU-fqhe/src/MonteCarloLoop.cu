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

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <exception>
#include <cmath>

#include <omp.h>
#include <cuda.h>
#include <mpi.h>

#include "MonteCarloLoop.h"
#include "LUCoreGPU.h"
#include "LUCoreCPU.h"
#include "PermHandler.h"
#include "complex.cuh"

#include "checks.cuh"

using namespace MPI;
using namespace std;


// Constructors
/* Class / namespace	:	MonteCarloLoop
 * Function		:	Default constructor
 * Inputs		:
 * Outputs		:
 * Description		:	Initializes all the containers and data to NULL and 0
 */
MonteCarloLoop::MonteCarloLoop() {
		theta = NULL;
		phi = NULL;
		k1 = 0.0;
		k2 = 0.0;
		r1 = 0.0;
		r2 = 0.0;
		detResult = NULL;

		Q = 0;
		mu = 0;
		nu = 0;
}

MonteCarloLoop::MonteCarloLoop(RunInfo rInfo, int nLoop) {
	int matSide = rInfo.getMatSide();
	this->nLoop = nLoop;

#ifdef __DPREC__
		theta	= new double [2*matSide];
		phi		= new double [2*matSide];
		k1		= 1.0;
		k2		= 1.0;
		r1		= 1.0;
		r2		= 1.0;
		detResult = NULL;
#else
		theta	= new float [2*matSide];
		phi		= new float [2*matSide];
		k1		= 1.0f;
		k2		= 1.0f;
		r1		= 1.0f;
		r2		= 1.0f;
		detResult = NULL;
#endif
		Q = (matSide-3)/2;
		mu = 1;
		nu = 1;
}

// Destructor
MonteCarloLoop::~MonteCarloLoop() {
	if(theta != NULL) {
		delete [] theta;
		theta = NULL;
	}
	if( phi != NULL) {
		delete [] phi;
		phi = NULL;
	}
	if( detResult != NULL) {
		delete [] detResult;
		detResult = NULL;
	}
}

/* Class / Namespace	: 	MonteCarloLoop
 * Function				:	perform
 * Inputs				:
 * - Pointer to Node object
 * - Pointer to RunInfo object
 * - Pointer to pHandler object
 * Outputs				:	None
 *
 * Description			:
 * Performs the Monte Carlo Loop. First read necessary info from the input objects, then calculate the work load balance between the GPUs and the CPUs.
 * It then allocates memory for each thread and GPU.
 * Launches an OpenMP parallel construct in which each assigned thread will allocate memory on one GPU, then iterate over the number of Monte Carlo iterations.
 * If the program has to perform with permutations, the node # 0 will calculate Phi and Theta, then broadcast them to all nodes. Matrices are then constructed from these angles and input permutations.
 * Otherwise, the "permutations" are the matrices to deal with.
 *
 *
 *
 */
void MonteCarloLoop::perform(Node *node, RunInfo *rInfo, PermHandler *pHandler, int mode) {

	//// Info from input objects 
	int nodCnt, cpuCnt, gpuCnt, rank, matSide, matNumElem, nElemPnode;

	try {
		nodCnt 	= node->getGlobalInfo()->getNodCnt();

		cpuCnt	= node->getLocalInfo()->getCpuCnt();
		gpuCnt 	= node->getLocalInfo()->getGpuInfo()->getGpuCnt();
		rank	= node->getLocalInfo()->getRank();
		matSide	= rInfo->getMatSide();
		matNumElem	= matSide*matSide;		// Number of elements per matrix
		nElemPnode	= rInfo->getNElemPnode();
	} catch(exception& e) {
		cerr << "Error \t Could not access stored information " << e.what() << endl;
		throw exception();
	}	

	//// Calculate how much work has to be sent to each GPU
	// Assume 32 CPU = 1 GPU
	float totalGpuCnt = (float)gpuCnt + (float)cpuCnt/32.f;
	int nMatGpu = (int)floor((float)nElemPnode/totalGpuCnt);	// Number of matrices per GPU
	int nMatCpu = nElemPnode - nMatGpu*gpuCnt;			// Number of matrices for ALL CPU

	if( nMatGpu % 2 != 0) {  	// If the number of matrices per GPU is not even,
		nMatGpu--;		// Then decrease the counter by 1
		nMatCpu += gpuCnt;	// which means that the CPU has to deal with <number of GPUs> more matrices
	}

#ifdef __VERBOSE__
	cout << "INFO \t Node " << rank << "\t Number of matrices for the node " << nElemPnode << " \t Number of matrices per GPU " << nMatGpu 	<<  " \t Number of matrices for the CPU(s) " << nMatCpu << endl;
#endif

	// Create the arrays
#ifdef __DPREC__
	// Holder for all the device arrays
	double2 **d_MatricesHolder = new double2* [gpuCnt];
	double2 **d_ReductionHolder = new double2* [gpuCnt];

	double2 *cpu_Matrices 	= NULL;//new double2 [nMatCpu*matNumElem];
	double2 *det_holder	= new double2 [gpuCnt+1];		// Holder for the results of the GPUs and the result of the CPUs

	double2 sum;
	sum.x = 0.0f;
	sum.y = 0.0f;

	double2	MPI_Final_Sum;
	MPI_Final_Sum.x = 0.0f;
	MPI_Final_Sum.y = 0.0f;
#else
	// Holder for all the device arrays
	float2 **d_MatricesHolder = new float2* [gpuCnt];
	float2 **d_ReductionHolder = new float2* [gpuCnt];

	float2 *cpu_Matrices 	= NULL; //new float2 [nMatCpu*matNumElem];
	float2 *det_holder	= new float2 [gpuCnt+1];		// Holder for the results of the GPUs and the result of the CPUs

	float2 sum;
	sum.x = 0.0f;
	sum.y = 0.0f;

	float2	MPI_Final_Sum;
	MPI_Final_Sum.x = 0.0f;
	MPI_Final_Sum.y = 0.0f;
#endif

	////  Send work to each thread: private variables
	cudaStream_t stream1;
	int th_num	= 0;

	#pragma omp parallel default(shared) private(th_num,stream1) 
	{
		th_num = omp_get_thread_num();
		// Allocate each GPU memory
		if(th_num < gpuCnt) {
#ifdef __VERBOSE__
			cout << "INFO \t Node " << rank << "\t Thread " << th_num << " allocating memory on GPU " << th_num << endl;
#endif			
			cudaSetDevice(th_num);
#ifdef __DPREC__
			cudaMalloc((void**)&d_MatricesHolder[th_num],2*nMatGpu*matNumElem*sizeof(double));
			cudaCheckError("Allocation of d_matrices");
			cudaMalloc((void**)&d_ReductionHolder[th_num],2*nMatGpu*matNumElem*sizeof(double));
			cudaCheckError("Allocation of d_matrices");
#else
			cudaMalloc((void**)&d_MatricesHolder[th_num],2*nMatGpu*matNumElem*sizeof(float));
			cudaCheckError("Allocation of d_matrices");
			cudaMalloc((void**)&d_ReductionHolder[th_num],nMatGpu*matNumElem*sizeof(float));
			cudaCheckError("Allocation of d_matrices");
#endif
		}
	}	
	srand(time(NULL));

	// Start the loop
	for (int j=0; j<nLoop; ++j){

		sum.x = 0.0f;
		sum.y = 0.0f;

		double begin,end,GPUbegin,GPUend;

/* If the permutations are defined:
 	 - create phi and theta
 	 - broadcast phi and theta
 	 - create matrices with mkMatrix

 	 Otherwise just perform an LU dec. with input matrices
*/
		// If permutations
		if( mode == PERM )
		{
			// Calculate Phi and Theta
			if(rank==0){
#ifdef __VERBOSE__
				cout << "INFO \t Calculating random configuration for MC loop " << j << " on node 0" << endl;
#endif
				begin = omp_get_wtime();
				#pragma omp parallel for default(shared)
				for (int i=0; i<2*matSide; i++){
	// TODO: random numbers for double prec.
#ifdef __DPREC__

#else
					theta[i]= PI * (float) rand() / (float) RAND_MAX / 2.0f;
					phi[i] 	= 2* PI * (float) rand() / (float) RAND_MAX / 2.0f;
#endif
				}
				end = omp_get_wtime();
#ifdef __VERBOSE__
				for (int i=0; i<2*matSide; i++)
					cout << "\t phi/2["<< i << "] " << phi[i] << "\t theta/2[" << i << "] " << theta[i] << endl;
#endif

#ifdef __VERBOSE__
				cout << "INFO \t Time to create Phi and Theta \t " << end-begin << " s" << endl;
#endif
			}

			// Send Phi and Psi across the network to all the nodes
			if (nodCnt>1){
#ifdef __DPREC__
				try {
					COMM_WORLD.Bcast(theta,matSide,MPI_DOUBLE,0);
					COMM_WORLD.Bcast(phi,matSide,MPI_DOUBLE,0);
				}
#else
				try {
					COMM_WORLD.Bcast(theta,matSide,MPI_FLOAT,0);
					COMM_WORLD.Bcast(phi,matSide,MPI_FLOAT,0);
					
				}
#endif
				catch(MPI::Exception& e) {
					cerr << "ERROR \t Node " << rank << " failed to broadcast / receive " << endl;
					throw exception();
				}
			}

			// TODO: Make matrices
#ifdef __VERBOSE__
			cout << "INFO \t Node " << rank << "\t Calling makeMatrixElements" << endl;
#endif
			begin = omp_get_wtime();
			//	makeMatrixElements(rank, hostname, phi,theta,k1,k2,r1,r2,mu,nu,Q,myPerms,devPermutations,devMatrices);
			end = omp_get_wtime();
#ifdef __VERBOSE__
			cout << "INFO \t Node " << rank << "\t Time for makeMatrixElements " << end-begin << endl;
#endif
		} 
		else {
			cpu_Matrices = (float2*)(pHandler->getLocalMatrices())+gpuCnt*nMatGpu*matNumElem;
		}

		cout << "INFO \t Node " << rank << "\t Calling LUCORE" << endl;
		// Set up the GPU
		GPUbegin = omp_get_wtime();
		begin = omp_get_wtime();
		#pragma omp parallel default(shared) private(th_num,stream1) 
		{
			th_num = omp_get_thread_num();
			#ifndef __NOGPU__
			if(th_num < gpuCnt) {
				// Set the device and stream
				cudaSetDevice(th_num);
#ifdef __DEBUG__				
				cudaCheckError("Set device inner loop");
#endif 				
				cudaStreamCreate(&stream1);
#ifdef __DEBUG__				
				cudaCheckError("Create stream inner loop");
#endif 				

				// Copy matrices to the device; Only for LU; For PERM, this is taken care of in makeMatrix
				if( mode == LU) {
					cudaMemcpyAsync(d_MatricesHolder[th_num],(float2*)(pHandler->getLocalMatrices())+th_num*nMatGpu*matNumElem,2*nMatGpu*matNumElem*sizeof(float),cudaMemcpyHostToDevice,stream1);
#ifdef __DEBUG__					
					cudaCheckError("Async memcpy to device");
#endif					
				}	
#ifdef __VERBOSE__
				cout << "INFO \t Node " << rank << "\t Calling LUCore GPU" << endl;
#endif				
				// Queue the GPU LU kernel and the global reduction
				LUCoreGPU::CALL(d_MatricesHolder,nMatGpu,rInfo,rank,th_num,&stream1);
				LUCoreGPU::globalReduction(d_MatricesHolder,d_ReductionHolder,nMatGpu,rank,th_num,stream1);

				// Copy back into the determinant holder
				cudaMemcpyAsync(&det_holder[th_num],&d_ReductionHolder[th_num][0],sizeof(float2),cudaMemcpyDeviceToHost,stream1);
#ifdef __DEBUG__				
				cudaCheckError("Async memcpy from device");
#endif				
			}
			#endif
		}		
		end = omp_get_wtime();

#ifdef __VERBOSE__
		cout << "INFO \t Node " << rank << "\t Time to queue GPU processing " << end-begin << endl;
		cout << "INFO \t Node " << rank << "\t Calling LUCore CPU" << endl;
#endif
		begin = omp_get_wtime();
		LUCoreCPU::CALL(cpu_Matrices,&det_holder[gpuCnt],nMatCpu,rInfo,rank,cpuCnt);
		end = omp_get_wtime();

#ifdef __VERBOSE__
		cout << "INFO \t Node " << rank << "\t Time to perform LUCore CPU " << end-begin << endl;
#endif
		#pragma omp parallel default(shared) private(th_num) 
		{
			th_num = omp_get_thread_num();
			// Wait for the GPU to finish its work
			if(th_num < gpuCnt) {
				cudaSetDevice(th_num);
				cudaDeviceSynchronize();
			}
			#pragma omp barrier
		}	
		GPUend=omp_get_wtime();

#ifdef __VERBOSE__
		cout << "INFO \t Node " << rank << "\t Time to perform LUCore CPU+GPU " << GPUend-GPUbegin << endl;
#endif

		// Perform the last reduction on the node
		for(int i = 0; i < gpuCnt+1; i++){
			 cout << "[ " << i << " ] = " << setprecision(15) << det_holder[i].x << " + I* " << det_holder[i].y << endl;
			 sum.x = sum.x+det_holder[i].x;
			 sum.y = sum.y+det_holder[i].y;
		 }
		cout << "RESULT \t Node " << rank << " \t Partial wf result \t" << setprecision (15) << sum.x << " + I " << sum.y << endl;

		if( nodCnt > 1 ) {
			try{
				// Reduce
				// Wait for all the nodes
				COMM_WORLD.Barrier();
#ifdef __DPREC__
				COMM_WORLD.Reduce(&sum.x,&(MPI_Final_Sum.x),1,MPI_DOUBLE,MPI_SUM,0);
				COMM_WORLD.Barrier();
				COMM_WORLD.Reduce(&sum.y,&(MPI_Final_Sum.y),1,MPI_DOUBLE,MPI_SUM,0);
#else
				COMM_WORLD.Reduce(&sum.x,&(MPI_Final_Sum.x),1,MPI_FLOAT,MPI_SUM,0);
				COMM_WORLD.Barrier();
				COMM_WORLD.Reduce(&sum.y,&(MPI_Final_Sum.y),1,MPI_FLOAT,MPI_SUM,0);
#endif
			}
			catch(MPI::Exception& e) {
				cerr << "ERROR \t Node " << rank << " failed to perform the final reduction " << endl;
				throw exception();
			}
		}
		else {
			MPI_Final_Sum.x = sum.x;
			MPI_Final_Sum.y = sum.y;
		}	

		if(rank == 0) {
				cout << "RESULT \t Node " << rank << " \t Final wf result \t" << MPI_Final_Sum.x << " + I " << MPI_Final_Sum.y << endl;
		}
	} // END OF THE MONTE CARLO LOOP

	// Free the GPU memory
	#pragma omp parallel default(shared) private(th_num,stream1)
	{
		th_num = omp_get_thread_num();
		if(th_num < gpuCnt ) {
			cudaFree(d_MatricesHolder[th_num]);
			cudaFree(d_ReductionHolder[th_num]);
		}
	}		

#ifdef __VERBOSE__
	cout << "INFO \t Node " << rank << "\t Freeing det_holder" << endl;
#endif
	delete [] det_holder;
	det_holder = NULL;

#ifdef __VERBOSE__
	cout << "INFO \t Node " << rank << "\t Freeing matricesHolder" << endl;
#endif
	delete [] d_MatricesHolder;
	d_MatricesHolder = NULL;

#ifdef __VERBOSE__
	cout << "INFO \t Node " << rank << "\t Freeing reductionHolder" << endl;
#endif
	delete [] d_ReductionHolder;
	d_ReductionHolder = NULL;
}


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



#include <cusp/coo_matrix.h>
#include <cusp/krylov/gmres.h>
#include <cusp/print.h>
#include <cusp/monitor.h>


#include <thrust/device_ptr.h>
#include <vector>
#include "main.h"
#include "checks.cuh"
using namespace std;


texture<float, 1, cudaReadModeElementType> k_tex;
texture<float, 1, cudaReadModeElementType> y_tex;

__global__ void eval_f(eval_node * f_nodes_dev, 
		int * terms_dev, 
		float * function_dev, 
		int * offset_terms_dev){

	__shared__ volatile float scratch[SHARED_BUF_SIZE];


	//could use constant mem here
	int index = offset_terms_dev[blockIdx.x];
	int terms_this_function = terms_dev[blockIdx.x];

	float fnt = 0.0f;

	if (threadIdx.x<terms_this_function){
		eval_node node = f_nodes_dev[index+threadIdx.x];

		fnt		=  node.constant;
		fnt		*= tex1Dfetch(k_tex, node.k_index-1);
		//zero based indexing
		fnt		*= powf(tex1Dfetch(y_tex, node.y_index_1-1),node.y_exp_1);	
		if (node.y_index_2 != -1)
			fnt		*= powf(tex1Dfetch(y_tex, node.y_index_2-1),node.y_exp_2);	

		//printf("b : %i t: %i c: %f k: %i y1: %i e1: %f y2: %i e2: %f fnt : %f tr : %f y: %f\n",\
		blockIdx.x,threadIdx.x,node.constant,node.k_index,node.y_index_1,node.y_exp_1,node.y_index_2,\
			node.y_exp_2, fnt, tex1Dfetch(k_tex,node.k_index-1), tex1Dfetch(y_tex, node.y_index_1-1));

	}

	scratch[threadIdx.x] = fnt;

	__syncthreads();

	if (blockDim.x >= 256){
		if (threadIdx.x < 128){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 128 ];
		}
		__syncthreads();
	}

	if (blockDim.x >= 128){
		if (threadIdx.x < 64){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 64 ];
		}
		__syncthreads();
	}

	if (blockDim.x >= 64){
		if (threadIdx.x < 32){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 32 ];
		}
	}


	if (threadIdx.x < 16) 		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 16 ];
	if (threadIdx.x < 8)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 8 ];
	if (threadIdx.x < 4)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 4 ];
	if (threadIdx.x < 2)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 2 ];
	if (threadIdx.x < 1)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 1 ];



	//we solve J*delta = -f
	if (threadIdx.x == 0){
		function_dev[blockIdx.x] 	= -scratch[0];
	}



}
__global__ void evalj(eval_node * j_nodes_dev, 
		int * terms_jac_dev, 
		float * jacobian_dev,
		int * offset_terms_jac_dev){			

	__shared__ volatile float scratch[SHARED_BUF_SIZE];



	//could use constant mem here
	int index = offset_terms_jac_dev[blockIdx.x];
	int terms_this_function = terms_jac_dev[blockIdx.x];
	float fnt = 0.0f;


	if (threadIdx.x<terms_this_function){



		eval_node node = j_nodes_dev[index+threadIdx.x];


		fnt		=  node.constant;
		int K_index	= (node.k_index-1);
		fnt		*= tex1Dfetch(k_tex, K_index);
		//zero based indexing
		if (node.y_exp_1 != 0)
			fnt		*= powf(tex1Dfetch(y_tex, node.y_index_1-1),node.y_exp_1);	
		if (node.y_index_2 != -1)
			fnt		*= powf(tex1Dfetch(y_tex, node.y_index_2-1),node.y_exp_2);	

		//if (blockIdx.x==0) printf("b : %i t: %i c: %f k: %i y1: %i e1: %f y2: %i e2: %f fnt : %f tr : %f y: %f\n",\
		blockIdx.x,threadIdx.x,node.constant,node.k_index,node.y_index_1,node.y_exp_1,node.y_index_2,\
			node.y_exp_2, fnt, tex1Dfetch(k_tex,node.k_index-1), tex1Dfetch(y_tex, node.y_index_1-1));

	}

	scratch[threadIdx.x] = fnt;

	__syncthreads();

	if (blockDim.x >= 256){
		if (threadIdx.x < 128){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 128 ];
		}
		__syncthreads();
	}

	if (blockDim.x >= 128){
		if (threadIdx.x < 64){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 64 ];
		}
		__syncthreads();
	}

	if (blockDim.x >= 64){
		if (threadIdx.x < 32){
			scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 32 ];
		}
	}


	if (threadIdx.x < 16) 		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 16 ];
	if (threadIdx.x < 8)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 8 ];
	if (threadIdx.x < 4)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 4 ];
	if (threadIdx.x < 2)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 2 ];
	if (threadIdx.x < 1)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 1 ];



	if (threadIdx.x == 0){


		jacobian_dev[blockIdx.x] 	= scratch[0];
	}

}

__global__ void update_y_from_delta(float * delta, float * y_dev, int f_size, float scale){

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if (gridDim.x==blockIdx.x){

		int thread_mask = (f_size - (blockIdx.x-1)*THREADS_UP);

		if (threadIdx.x < thread_mask){
			y_dev[index] += delta[index] / scale;

		}
	} else{

		y_dev[index] += delta[index] / scale;
	}

}

__global__ void prep_function_L2(float * function_dev, int f_size){

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if (gridDim.x==blockIdx.x){

		int thread_mask = (f_size - (blockIdx.x-1)*THREADS_UP);

		if (threadIdx.x < thread_mask){
			function_dev[index] *= function_dev[index];

		}
	} else{

		function_dev[index] *= function_dev[index];
	}

}



int solve(vector<int>& index_i, 	//index i
		vector<int>& index_j,		//index j
		vector<float>& k_data,		//actual k data to load
		float * function_dev,		//function values on device
		float * delta,			//the delta we calculate every iteration of J*delta = -f
		float * jacobian_dev,		//the jacobian values
		float tol,			//the desired tolerance in L1/L2 norm
		eval_node * j_nodes_dev,	//how to evaluate each jac term
		int * terms_jac_dev,		//how many terms/nodes/structs are in each jac eval
		int * offset_terms_jac_dev,			
		eval_node * f_nodes_dev,	//how to evaluate each function
		int * terms_dev,		//how many terms/nodes/structs are in each f eval
		int * offset_terms_dev,
		int max_terms_j,		//the max terms in each J(i,j)
		int max_terms_f,		// " f 
		int j_size,			//total J(i,j)
		int f_size,			//total f's
		float * y_dev,			//solution on device
		float * output){	


	//scale for updates
	float scale = 10.0f;

	//get k data onto device, create textures
	int size_k = k_data.size();
	float * tmp_k = new float[size_k];

	for (int i=0; i<size_k; i++){
		tmp_k[i] = k_data[i];
	}

	float *kArray = NULL;
	cudaMalloc((void**) &kArray, sizeof(float)*size_k);
	cudaCheckError("malloc, kArray");
	cudaMemcpy(kArray, tmp_k, sizeof(float)*size_k, cudaMemcpyHostToDevice);
	cudaCheckError("memcpy, kArray");

	// Bind textures
	cudaBindTexture(0, k_tex, kArray,size_k*sizeof(float));
	cudaCheckError("bindTex, k");

	cudaBindTexture(0, y_tex, y_dev, f_size*sizeof(float));
	cudaCheckError("bindTex, y");


	//set-up kernel configs
	dim3 blocks_j, blocks_f, blocks_up, blocks_l2, threads_j, threads_f, threads_up, threads_l2;

	blocks_j.x 	= j_size; 
	threads_j.x	= (max_terms_j < 32) ? 32 : ceil((float) max_terms_j / 32.0f) * 32;

	blocks_f.x 	= f_size;
	threads_f.x	= (max_terms_f < 32) ? 32 : ceil((float) max_terms_f / 32.0f) * 32;

	blocks_up.x	= f_size / THREADS_UP +1;
	threads_up.x	= THREADS_UP;

#ifdef __DEBUG
	cerr << "blocks/threads for jac " << blocks_j.x << " " << threads_j.x << endl;
	cerr << "blocks/threads for func " << blocks_f.x << " " << threads_f.x << endl;
	cerr << "blocks/threads for up " << blocks_up.x << " " << threads_up.x << endl;
#endif

	// create sparse matrix structure (COO format)
	cusp::coo_matrix<int, float, cusp::device_memory> J(f_size,f_size,j_size);

	//get indices setup
	thrust::copy(index_i.begin(), index_i.end(), J.row_indices.begin());
	thrust::copy(index_j.begin(), index_j.end(), J.column_indices.begin());

	// setup delta & rhs
	cusp::array1d<float, cusp::device_memory> d(J.num_rows, 0);
	cusp::array1d<float, cusp::device_memory> f(J.num_rows, 0);

	//cast raw pointers to thrust objects
	thrust::device_ptr<float> dp = thrust::device_pointer_cast(delta);
	thrust::device_ptr<float> jp = thrust::device_pointer_cast(jacobian_dev);
	thrust::device_ptr<float> fp = thrust::device_pointer_cast(function_dev);
	//thrust::device_ptr<float> yp = thrust::device_pointer_cast(y_dev);

	//best y to date, for restarts
	//thrust::device_vector<float> besty(f_size);
	//thrust::copy_n(yp, f_size, besty.begin());


	//test J
	evalj<<<blocks_j,threads_j>>>(j_nodes_dev, 
			terms_jac_dev, 
			jacobian_dev, 
			offset_terms_jac_dev);			


	//copy back initial J and see how we're doing

	float * test_J = new float[j_size];
	cudaMemcpy(test_J, jacobian_dev, sizeof(float)*j_size, cudaMemcpyDeviceToHost);

	float val = 0.0f;
	int index = 0;
	for (int i=0; i<index_i.size(); i++){

		if (index_i[i] > index){
			index++;

			if (val==0.0f){
				cerr << "all zeros along Jacobian row index : " << index-1 << endl;
				throw std::invalid_argument( "singular matrix" );

			}			

			val = 0.0f;

		} else{

			val += test_J[i];

		}

	}

	delete[] test_J;
	//	calculate an initial f

	eval_f<<<blocks_f,threads_f>>>(f_nodes_dev, 
			terms_dev, 
			function_dev, 
			offset_terms_dev);
	cudaThreadSynchronize();	

	cudaCheckError("eval_f");

	//prep_function_L2<<<blocks_up, threads_up>>>(function_dev, f_size);
	//cudaCheckError("prep_function_L2");

	//float best_L2 = thrust::reduce(fp, fp+f_size);


	//Newton-Raphson iterations
	for (int i=0; i<MAX_ITERATIONS; i++){

		//cusp::print(J);

		//update f

		eval_f<<<blocks_f,threads_f>>>(f_nodes_dev, 
				terms_dev, 
				function_dev, 
				offset_terms_dev);
		cudaThreadSynchronize();	

		cudaCheckError("eval_f");

		thrust::copy_n(fp, f_size, f.begin());

#ifdef __VERBOSE
		cusp::verbose_monitor<float> monitor(f, 5000, 1e-8);
#else
		cusp::default_monitor<float> monitor(f, 5000, 1e-8);
#endif
		// solve the linear system J * d = -f 
		cudaThreadSynchronize();	
		cusp::krylov::gmres(J, d, f,500,monitor); 

		cudaThreadSynchronize();	
		thrust::copy(d.begin(), d.end(), dp);
		update_y_from_delta<<<blocks_up, threads_up>>>(delta, y_dev, f_size,scale);
		cudaCheckError("update_y");
		prep_function_L2<<<blocks_up, threads_up>>>(function_dev, f_size);
		cudaCheckError("prep_function_L2");

		cudaThreadSynchronize();	
		float f1 = thrust::reduce(fp, fp+f_size);


		if(fpe(f1)){
			throw std::invalid_argument( "encountered fpe in solve" );
		}

		cout << "iteration : " << i << " equations L2 norm : " << f1 << endl;

		if (f1 < tol)
			break;


		//if (f1 <= best_L2){

		//	cerr << "first branch" << endl;

		//	thrust::copy_n(yp, f_size, besty.begin());
		//	cudaThreadSynchronize();	
		//	best_L2 = f1;
		//calculate y+=d
		//	thrust::copy(d.begin(), d.end(), dp);
		//	update_y_from_delta<<<blocks_up, threads_up>>>(delta, y_dev, f_size,scale);
		//	cudaCheckError("update_y_from_delta");

		//	cudaThreadSynchronize();	
		//} else {

		//	cerr << "second branch" << endl;
		//adaptive steps; back it up to best y, scale the delta
		//	thrust::copy(besty.begin(), besty.end(), yp);

		//	scale *= 2.0;

		//	cerr << "scale " << scale << endl;
		//	cudaThreadSynchronize();	

		//for (int i=0; i<besty.size(); i++)
		//	cout << besty[i] << endl;
		//}

		//update J
		evalj<<<blocks_j,threads_j>>>(j_nodes_dev, 
				terms_jac_dev, 
				jacobian_dev, 
				offset_terms_jac_dev);			

		cudaThreadSynchronize();	
		cudaCheckError("eval_j");
		thrust::copy_n(jp, j_size, J.values.begin());


	}


	//final equations/functions
	eval_f<<<blocks_f,threads_f>>>(f_nodes_dev, 
			terms_dev, 
			function_dev, 
			offset_terms_dev);
	cudaThreadSynchronize();	

	cudaCheckError("eval_f");

	cerr << "FINAL FUNCTIONS FOR THIS SOLN" << endl;
	cusp::print(f);

	cudaMemcpy(output, y_dev, sizeof(float)*f_size, cudaMemcpyDeviceToHost);

	cudaFree(function_dev);		//function values on device
	cudaFree(delta);		//the delta we calculate every iteration of J*delta = -f
	cudaFree(jacobian_dev);		//the jacobian values
	cudaFree(j_nodes_dev);		//how to evaluate each jac term
	cudaFree(terms_jac_dev);	//how many terms/nodes/structs are in each jac eval
	cudaFree(f_nodes_dev);		//how to evaluate each function
	cudaFree(terms_dev);		//how many terms/nodes/structs are in each f eval
	cudaFree(y_dev);		//solution
	cudaFree(kArray);

	cudaUnbindTexture(k_tex);
	cudaUnbindTexture(y_tex);

	return 0;
}

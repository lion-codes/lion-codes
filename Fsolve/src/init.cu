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
#include <vector>
#include <map>
#include <ctime>

#include "main.h"
//#include "checks.cuh"

using namespace std;

__global__ void print_Y(float *Y) {
	for(int i = 0;i<179;i++)
		printf("Y[%d] = %e\n",i,Y[i]);

}		

void init_f(vector<float> &k_inds, 			//as before
	vector<float>& constants, 			//
	vector<map<float,float> >& y_complete, 		//
	vector<int>& terms, 				//
	eval_node ** f_nodes_dev, 			//all the node data in an array, device pointer
	int ** terms_dev,				//the term information, device pointer
	int ** offset_terms_dev,			//offsets for the term information, device pointer
	float ** function_dev,				//storage for f(y)
	float ** delta,					//storage for delta
	float ** y_dev,					//solution vector
	vector<float>& iv){					//init guess


	int num_leaves 	= constants.size();
	int num_funcs 	= terms.size();
	
	eval_node * tmp_nodes 		= new eval_node[num_leaves ];
	int * tmp_terms 		= new int[num_funcs ];
	float * tmp_y	 		= new float[num_funcs ];
	int * tmp_offsets	 	= new int[num_funcs ];

	tmp_offsets[0]	= 0;
	int off 	= terms[0];

	for (int i=1; i<num_funcs; i++){
		tmp_offsets[i] = off;
		off+=terms[i];

	}

	for (int i=0; i<num_funcs; i++)
		tmp_terms[i] = terms[i];

	for (int i=0; i<num_leaves; i++){
		tmp_nodes[i].constant 	= constants[i];
		tmp_nodes[i].k_index 	= (int) k_inds[i];
		tmp_nodes[i].y_index_1	= 0;
		tmp_nodes[i].y_exp_1	= 1.0;
		tmp_nodes[i].y_index_2	= -1;
		tmp_nodes[i].y_exp_2	= 1.0;
	
		map<float,float> tmp = y_complete[i];

		tmp_nodes[i].y_index_1 	= (int) tmp.begin()->first;
		tmp_nodes[i].y_exp_1 	= tmp.begin()->second;
	
		if (tmp.size()>1){

		map<float,float>:: iterator it = tmp.begin();
		it++;
		tmp_nodes[i].y_index_2 	= (int) it->first;
		tmp_nodes[i].y_exp_2 	= it->second;

		}
	}


	cudaMalloc(f_nodes_dev, sizeof(eval_node)*num_leaves);

	//cudaCheckError("malloc, f_nodes_dev");
	cudaMemcpy(*f_nodes_dev, tmp_nodes, sizeof(eval_node)*num_leaves, cudaMemcpyHostToDevice);
	//cudaCheckError("memcpy, f_nodes_dev");
	cudaMalloc(terms_dev, sizeof(int)*num_funcs);
	cudaMalloc(offset_terms_dev, sizeof(int)*num_funcs);
	//cudaCheckError("malloc, terms_dev");
	cudaMemcpy(*terms_dev, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
	cudaMemcpy(*offset_terms_dev, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
	//cudaCheckError("memcpy, terms_dev");
	cudaMalloc(function_dev, sizeof(float)*num_funcs);
	//cudaCheckError("malloc, function_dev");
	cudaMalloc(delta, sizeof(float)*num_funcs);
	//cudaCheckError("malloc, delta");
	

	//cout << num_funcs << endl;
	//init y
//	srand(time(NULL));
	srand(1024);
	for (int i=0; i<num_funcs; i++)
		tmp_y[i] = iv[i]; //guess * rand() / (float) RAND_MAX;


	cudaMalloc(y_dev, sizeof(float)*num_funcs);
	//cout << "ydev 2 " << *y_dev << endl;
	//cudaCheckError("malloc, y_dev");
	cudaMemcpy(*y_dev, tmp_y, sizeof(float)*num_funcs, cudaMemcpyHostToDevice);
	//cudaCheckError("memcpy, y_dev");

	delete[] tmp_terms, tmp_nodes, tmp_y, tmp_offsets;
}



void init_j(vector<float> &k_inds_jac, 			//as before
	vector<float>& constants_jac, 			//
	vector<map<float,float> >& jac_complete, 	//
	vector<int>& terms_jac, 			//
	eval_node ** jac_nodes_dev, 			//all the node data in an array, device pointer
	int ** terms_jac_dev,				//the term information, device pointer
	int ** offset_terms_jac_dev,			//offset for the term information, device pointer
	float ** jacobian_dev){				//storage for the results J(y)


	int num_leaves 	= constants_jac.size();
	int num_funcs 	= terms_jac.size();
	
	eval_node * tmp_nodes 	= new eval_node[num_leaves ];
	int * tmp_terms 		= new int[num_funcs ];
	int * tmp_offsets	 	= new int[num_funcs ];

	tmp_offsets[0]	= 0;
	int off 	= terms_jac[0];

	for (int i=1; i<num_funcs; i++){
		tmp_offsets[i] = off;
		off+=terms_jac[i];

	}


	for (int i=0; i<num_funcs; i++)
		tmp_terms[i] = terms_jac[i];

	for (int i=0; i<num_leaves; i++){
		tmp_nodes[i].constant 	= constants_jac[i];
		tmp_nodes[i].k_index 	= (int) k_inds_jac[i];
		tmp_nodes[i].y_index_1	= 0;
		tmp_nodes[i].y_exp_1	= 1.0;
		tmp_nodes[i].y_index_2	= -1;
		tmp_nodes[i].y_exp_2	= 1.0;
	
		map<float,float> tmp = jac_complete[i];

		tmp_nodes[i].y_index_1 	= (int) tmp.begin()->first;
		tmp_nodes[i].y_exp_1 	= tmp.begin()->second;
		
		if (tmp.size()>1){

		map<float,float>:: iterator it = tmp.begin();
		it++;
		tmp_nodes[i].y_index_2 	= (int) it->first;
		tmp_nodes[i].y_exp_2 	= it->second;

		}
	}


	cudaMalloc(jac_nodes_dev, sizeof(eval_node)*num_leaves);
	//cudaCheckError("malloc, jac_nodes_dev");
	cudaMemcpy(*jac_nodes_dev, tmp_nodes, sizeof(eval_node)*num_leaves, cudaMemcpyHostToDevice);
	//cudaCheckError("memcpy, jac_nodes_dev");
	


	cudaMalloc(terms_jac_dev, sizeof(int)*num_funcs);
	cudaMalloc(offset_terms_jac_dev, sizeof(int)*num_funcs);
	//cudaCheckError("malloc, terms_jac_dev");
	cudaMemcpy(*terms_jac_dev, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
	cudaMemcpy(*offset_terms_jac_dev, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
	//cudaCheckError("memcpy, terms_jac_dev");
	cudaMalloc(jacobian_dev, sizeof(float)*num_funcs);
	//cudaCheckError("malloc, jacobian_dev");


	delete[] tmp_terms, tmp_nodes, tmp_offsets;


}

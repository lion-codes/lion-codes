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



#include "stdio.h"
#include "stdlib.h"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <assert.h>
#include <cuda.h>

#ifndef __MAIN_H
#define __MAIN_H

#define SHARED_BUF_SIZE 256
#define MAX_ITERATIONS 100
#define THREADS_UP 512
#define THREADS_L2 512
#define fpe(x) (isnan(x) || isinf(x))

//#define __VERBOSE
//#define __DEBUG


using namespace std;
void parser_main(char * filename, 			//csv file with k data
	vector<float>& k_data,				//k data itself
	vector<float>& k_inds, 				//the k indices for equations
	vector<float>& constants, 			//all constant/sign data
	vector<map<float,float> >& y_complete, 		//all y data (yindex(key) & power(value))
	vector<int>& terms, 				//the number of terms per function evaluation
	int& max_elements,				//the maximum number of terms in equation evaluation
	int& max_term_size);				//maximum elements encountered in equation term, for node size

void jacobian(vector<float>& k_inds, 			//input: the k indices for equations
	vector<float>& constants, 			//input: 	all constant/sign data
	vector<map<float,float> >& y_complete, 		//input: 	all y data (yindex(key) & power(value))
	vector<int>& terms, 				//input: 	the number of terms per function evaluation
	vector<int>& terms_jac, 			//output: 	the number of terms per jacobian i,j evaluation
	vector<int>& index_i,				//output: 	all the Jacobian row indices
	vector<int>& index_j,				//output:	all the non-zero Jacobian column indices
	vector<float>& k_inds_jac, 			//output:	the k indices for each jac term
	vector<float>& constants_jac, 			//output:	all constant/sign data for each jac term
	vector<map<float,float> >& jac_complete, 	//output:	all y data (yindex(key) & power(value)) for each jac term
	int& max_elements);				//output:	the maximum number of terms in equation evaluation


typedef struct{

	float constant;
	int k_index;
	int y_index_1;
	float y_exp_1;
	int y_index_2;
	float y_exp_2;

} eval_node;


void init_f(vector<float> &k_inds, 			//as before
	vector<float>& constants, 			//
	vector<map<float,float> >& y_complete, 		//
	vector<int>& terms, 				//
	eval_node ** f_nodes_dev, 			//all the node data in an array, device pointer
	int ** terms_dev,				//the term information, device pointer
	int ** offset_terms_dev,			//offsets for term information, device pointer
	float ** function_dev,				//storage for f(y)
	float ** delta,					//storage for delta
	float ** y_dev,					//storage for y, solution
	float guess);

void init_j(vector<float> &k_inds_jac, 			//as before
	vector<float>& constants_jac, 			//
	vector<map<float,float> >& jac_complete, 	//
	vector<int>& terms_jac, 			//
	eval_node ** jac_nodes_dev, 			//all the node data in an array, device pointer
	int ** terms_jac_dev,				//the term information, device pointer
	int ** offset_terms_jac_dev,			//offsets for the term information, device pointer
	float ** jacobian_dev);				//storage for the results J(y)

int solve(vector<int>& index_i, 			//i indices in COO rep
	vector<int>& index_j,				//j indices in COO rep
	vector<float>& k_data,				//actual k values
	float * function_dev,				//function values on device
	float * delta,					//delta, in J*delta = -f(y)
	float * jacobian_dev,				//jacobian values on device
	float tol,					//desired tolerance in solution to f(y)=0
	eval_node * j_nodes_dev,			//all the j nodes
	int * terms_jac_dev,				//the num. terms in each J(i,j) evaluation
	int * offset_terms_jac_dev,			//offsets for the term information, device pointer
	eval_node * f_nodes_dev,			//all the f nodes
	int * terms_dev,				//the num. terms in each f eval
	int * offset_terms_dev,				//offsets for term information, device pointer
	int max_terms_j,				//maximum terms in J eval
	int max_terms_f,				//max terms in f
	int j_size,					//the total size of J(i,j) sparse elements
	int f_size,					//the total size of f(y)
	float * y_dev,					//solution vector, device
	float * output);


void check_y(vector<map<float,float> >& y_complete);	//do just that

#endif

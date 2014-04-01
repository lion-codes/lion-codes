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


// a GPU based Fsolve
// WJB 08/13

using namespace std;

int main(int argc, char * argv[])
{


	if (argc < 4){
		cerr << "USAGE : fsolve.x <tolerance> <file with k data, comma separated> <soln. guess> & equations supplied on stdin" << endl;
		throw std::invalid_argument( "Received bad equation format on stdin" );
	}

	//all the data we read in, for function evaluation
	vector<float> k, k_inds, constants;
	vector<map<float,float> > y_complete;	

	//how many terms per equation
	vector<int> terms;


	//the maximum number of terms; this will help determine block size in GPU function evaluation
	int max_term_f=0;
	int max_term_j=0;
	int max_term_size=0;

	float tol = atof(argv[1]);
	float guess = atof(argv[3]);
	
	

	parser_main(argv[2], 
		k, 
		k_inds, 
		constants, 
		y_complete, 
		terms, 
		max_term_f, 
		max_term_size);

	cerr << "max_term & max_term_size" << endl;
	cerr << max_term_f << " " << max_term_size << endl;

	//check and maybe relabel y-indices
	check_y(y_complete);

	//we're only using ~ float6's for now ie., max of two y variables per term
	assert(max_term_size <= 2);

	int num_leaves_func = constants.size();

	//jacobian evaluation nodes generation
	//
	vector<int> terms_jac; 				//output: 	the number of terms per jacobian i,j evaluation
	vector<int> index_i;				//output: 	all the Jocobian row indices
	vector<int> index_j;				//output:	all the non-zero Jacobian column indices
	vector<float> k_inds_jac; 			//output:	the k indices for each jac term
	vector<float> constants_jac; 			//output:	all constant/sign data for each jac term
	vector<map<float,float> > jac_complete; 	//output:	all y data (yindex(key) & power(value)) for each jac term
	
	//get the evaluation nodes & a coordinate representation of the sparse matrix
	jacobian(k_inds, 		//k index
		constants, 		//constants/signs
		y_complete, 		//all ydata (base/exponents) for j
		terms, 			//the number of terms per function eval
		terms_jac, 		//output terms per J(i,j) eval
		index_i, 		//index i for same
		index_j, 		//index j for same
		k_inds_jac, 		//k indices for each i,j
		constants_jac, 		//ditto
		jac_complete, 		//all the y data for J
		max_term_j);		//maximum number of terms in J(i,j) eval

	int num_leaves_jac = constants_jac.size();
	
	//init
	eval_node * f_nodes_dev, *j_nodes_dev;	
	int * terms_dev, * terms_jac_dev;
	int * offset_terms_dev, * offset_terms_jac_dev;
	float * function_dev, *delta, *jacobian_dev, *y_dev, *output;


	output = new float[terms.size()];


	//ready the device for function eval
	init_f(k_inds, 
		constants, 
		y_complete, 
		terms, 
		&f_nodes_dev, 
		&terms_dev,
		&offset_terms_dev,
		&function_dev, 
		&delta,
		&y_dev,
		guess);
	
	//ready the device for jacobian eval	
	init_j(k_inds_jac, 
		constants_jac, 
		jac_complete,
		terms_jac, 
		&j_nodes_dev, 
		&terms_jac_dev,	
		&offset_terms_jac_dev,
		&jacobian_dev);


	//solve
	solve(index_i, 
		index_j, 
	   	k, 
		function_dev, 
		delta, 
		jacobian_dev, 
		tol, 
		j_nodes_dev, 
		terms_jac_dev, 
		offset_terms_jac_dev,			
		f_nodes_dev, 
		terms_dev, 
		offset_terms_dev,
		max_term_j, 
		max_term_f,
		index_i.size(),
		terms.size(),
		y_dev,
		output);


	cout << "Output Solution:" << endl;
	for (int i=0; i<terms.size(); i++)
		cout << output[i] << " " << endl;

	cout << endl;

	delete[] output;
	return 0;
}

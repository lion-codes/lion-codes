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



#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <assert.h>


using namespace std;


void jacobian(vector<float>& k_inds, 			//input: the k indices for equations
		vector<float>& constants, 			//input: 	all constant/sign data
		vector<map<float,float> >& y_complete, 		//input: 	all y data (yindex(key) & power(value))
		vector<int>& terms, 				//input: 	the number of terms per function evaluation
		vector<int>& terms_jac, 			//output: 	the number of terms per jacobian i,j evaluation
		vector<int>& index_i,				//output: 	all the Jocobian row indices
		vector<int>& index_j,				//output:	all the non-zero Jacobian column indices
		vector<float>& k_inds_jac, 			//output:	the k indices for each jac term
		vector<float>& constants_jac, 			//output:	all constant/sign data for each jac term
		vector<map<float,float> >& jac_complete, 	//output:	all y data (yindex(key) & power(value)) for each jac term
		int& max_elements){				//output: 	the maximum number of terms in any i,j evaluation

	int displacement=0;

	max_elements = -1;

	for (int i=0; i<terms.size(); i++){


		int j_size=terms[i];

		//gather the total y for an equation (key), don't care about vals
		//we use a std::map, exploiting insertion order
		map<float,vector<int> > tmp;

		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			for (map<float,float>::iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){
				tmp[itm->first].push_back(j);

			}

		}

		// these are all the valid j indices for this row (i) in jacobian
		for (map<float,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){
			//convert 1 based index to 0
			index_j.push_back((int) itmv->first -1);
			index_i.push_back(i);
		}

		//each GPU block is going to eval one i,j in jacobian
		//so as with function evaluation we'll store the number of terms in i,j evaluation
		//and thus a block will know how many nodes/structs to load

		for (map<float,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){

			//find this y in all the maps (one map per term)
			float y = itmv->first;
			int jac_terms=0;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				bool found=false;
				map<float,float> relevant_y;
				float pow = 1.0f;

				for (map<float,float>::iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){

					if (itm->first==y){
						found=true;
						pow = itm->second;				
					} else{
						relevant_y[itm->first] = itm->second;
					}
				}

				// if y appeared in this map, we have a non-zero contribution to the jacobian term
				if (found){
					jac_terms++;
					if (pow == 1.0f)
						constants_jac.push_back(constants[index]);
					else

						constants_jac.push_back(constants[index]*pow);


					k_inds_jac.push_back(k_inds[index]);

					if (pow != 1.0f){
						relevant_y[y]=pow-1;

					}

					jac_complete.push_back(relevant_y);			


				}

			}


			max_elements = (jac_terms > max_elements) ? jac_terms : max_elements;
			terms_jac.push_back(jac_terms);
		}

		displacement += terms[i];

	}

	assert(terms_jac.size()==index_j.size());
	//
	//

#ifdef __VERBOSE

	displacement =0;

	for (int i=0; i<terms_jac.size(); i++){


		cerr << "jac element : " << i << endl;
		cerr << "indices : " << index_i[i] << " " << index_j[i] << endl;
		int j_size=terms_jac[i];

		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			for (map<float,float>::iterator itm = jac_complete[index].begin(); itm != jac_complete[index].end(); itm++){
				cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

			}

		}
		cerr << "k_inds" << endl;
		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			cerr << k_inds_jac[index] << " ";

		}
		cerr << endl;
		cerr << "consts" << endl;
		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			cerr << constants_jac[index] <<  " ";

		}
		cerr << endl;

		displacement += terms_jac[i];

	}

#endif

}

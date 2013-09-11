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



#include <vector>
#include <map>
#include <iostream>

using namespace std;

void check_y(vector<map<float,float> >& y_complete){


	map<float,float> indices;

	//check and possibly relabel y indices
	//
	//
	for (vector<map<float, float> >::iterator it = y_complete.begin(); it != y_complete.end(); it++){

		map<float,float> y_eqs = *it;

		for (map<float,float>::iterator it1 = y_eqs.begin(); it1 != y_eqs.end(); it1++){
			indices.insert(make_pair(it1->first,0));

		}

	}


	int index=1; bool reorder =false;
	for (map<float,float>::iterator it = indices.begin(); it != indices.end(); it++, index++){

		it->second = index;


		if (index != it->first){

			cerr << "CAUTION; remapping y index " << it->first << " to " << index << endl;
			reorder=true;
		}

	}


	if (reorder){
		for (vector<map<float,float> >::iterator it=y_complete.begin(); it != y_complete.end(); it++){


				map<float,float> eq_term = *it; //= indices[it1->first];
				map<float,float> new_term;
		
				for (map<float,float>::iterator it1=eq_term.begin(); it1!=eq_term.end(); it1++){
					
					float tmp = indices[it1->first];

					new_term[tmp] = it1->second;
				}


				*it = new_term;

			}

		}


}

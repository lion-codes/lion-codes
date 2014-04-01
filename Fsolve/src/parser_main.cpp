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



#include "clientp.hpp"
#include "main.h"

using namespace clientp;

void parser_main(char * filenameA, 			//csv file with k data
		char * filenameB,			//init values file
		vector<float>& iv,			// init vales
		vector<float>& k,				//k data
		vector<float>& k_inds, 				//the k indices for equations
		vector<float>& constants, 			//all constant/sign data
		vector<map<float,float> >& y_complete, 		//all y data (yindex(key) & power(value))
		vector<int>& terms, 				//the number of terms per function evaluation
		int& max_elements,				//the maximum number of terms in equation evaluation
		int& max_term_size){				//maximum elements encountered in equation term, for node size

	ifstream ifile(filenameA);
	string value;
	bool success=false;

	while(getline(ifile,value)){
		success |= clientp::parse_csv(value.begin(), value.end(), k);
	}

	ifile.close();
	// abort;
	if (!success)
		throw std::invalid_argument( "loading of k data failed" );

	//initial values
	ifstream ivfile(filenameB);
	if (ivfile.good()){
		success=false;

		while(getline(ivfile,value)){
			success |= clientp::parse_csv(value.begin(), value.end(), iv);
		}

		ivfile.close();

		// abort;
		if (!success)
			throw std::invalid_argument( "loading of inital value data failed" );

	}


	//buffer for equations arriving on stdin & line num
	string input;
	int line_no=0;

	//what we do and don't want on stdin
	string unsup[]={"cosh","sinh","tanh","cos","sin","tan"};
	string sup[]={"+","-"};


	//maximum elements in terms, in order to decide size of struct for node storage
	max_elements=INT_MIN;
	max_term_size=INT_MIN;

	while(getline(cin, input)){

		unsigned count = 0;

		//c++0x
		//for (string &x : unsup)
		for (int i=0; i<6; i++)
			parse(input.c_str(),*(  str_p(unsup[i].c_str()) [ increment_a(count) ] | anychar_p ));

		// abort;
		if (count > 0){
			cerr << "Input line : " << line_no << endl; 
			cerr << "Received : " << input << endl; 
			throw std::invalid_argument( "Contains one or more unsupported functions \
					(*polynomials only please*) on stdin" );
		}


		count=0;
		//c++0x
		//for (string &x : sup)
		for (int i=0; i<2; i++)
			parse(input.c_str(),*(  str_p(sup[i].c_str()) [ increment_a(count) ] | anychar_p ));

		// abort;
		if (!(count > 0)){
			sad_little_message(line_no,input);
		}


		//tokenize just a little bit
		int index=0;
		string::iterator it;
		vector<string>::iterator its;
		string tmp_string="";
		vector<string> tokens;

		for ( it = input.begin() ; it < input.end(); it++,index++){


			if (((*it=='-') || (*it=='+')) && (index !=0)){
				tokens.push_back(tmp_string);
				tmp_string.clear();
				tmp_string+=*it;
			} else {

				tmp_string+=*it;
			}
		}

		tokens.push_back(tmp_string);

		index=0;
		int first_ele=0;
		float sign=1.0;

		for ( its = tokens.begin() ; its != tokens.end(); its++, index++){


			map<float,float> tmp;
			bool success = clientp::parse_constants(its->begin(), its->end(), constants);
			assert(constants.size());
			success |= clientp::parse_k_vals(its->begin(), its->end(), k_inds);
			assert(k_inds.size());
			success |= clientp::parse_y_vals(its->begin(), its->end(), tmp);



			if (!success)
				sad_little_message(line_no,*its);

			int sz = tmp.size();
			max_term_size = (sz > max_term_size) ? sz : max_term_size;

			y_complete.push_back(tmp);


		}

		max_elements = (index > max_elements) ? index : max_elements;
		//tally terms in this equation
		terms.push_back(index);

		line_no++;
	}


	assert ((y_complete.size()==constants.size()) &&( k_inds.size()==constants.size()));

#ifdef __VERBOSE

	int displacement =0;

	for (int i=0; i<terms.size(); i++){


		cerr << "line : " << i << endl;
		int j_size=terms[i];

		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			for (map<float,float>::iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){
				cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

			}

		}
		cerr << "k_inds" << endl;
		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			cerr << k_inds[index] << " ";

		}
		cerr << endl;
		cerr << "consts" << endl;
		for (int j=0; j<j_size; j++){

			int index = j+displacement;
			cerr << constants[index] <<  " ";

		}
		cerr << endl;



		displacement += terms[i];

	}

#endif


}

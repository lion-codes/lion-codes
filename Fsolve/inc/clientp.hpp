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



#include <boost/config/warning_disable.hpp>
#include <boost/bind.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_increment_actor.hpp>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <stdexcept>


#ifndef __CLIENTP_HPP
#define __CLIENTP_HPP

using namespace std;
using namespace BOOST_SPIRIT_CLASSIC_NS;


namespace clientp
{

	void sad_little_message(int line_no, string input){
		cerr << "Input line : " << line_no << endl; 
		cerr << "Received : " << input << endl; 
		cerr << "Please express equations using arithmetic combinations +/- of one or more terms, " << endl;
		cerr << "terms consisting of products expressed with *, powers with ^ and powers/indices" << endl;
		cerr << "expressed within parenthesis for constants k and variables y (1-based index) eg.," << endl;
		cerr << " " << endl;
		cerr << "1.0*k(717)*y(516)-1.0*k(416)*y(1)*y(392)-1.0*k(718)*y(1)^(1/2)*y(517)" << endl;
		cerr << " " << endl;
		throw std::invalid_argument( "Received bad equation format on stdin" );
	}



	void update(std::map<float,float> &x, const float& key, const float& val){

		x[key]+=val;
	}


	namespace qi = boost::spirit::qi;
	namespace ascii = boost::spirit::ascii;
	namespace phoenix = boost::phoenix;

	using qi::float_;
	using qi::phrase_parse;
	using qi::_1;
	using ascii::space;
	using phoenix::push_back;



	template <typename Iterator>
		bool parse_csv(Iterator first, Iterator last, std::vector<float>& k)
		{

			bool r = phrase_parse(first, last,

					//  grammar for csv files
					(
					 //*(float_ >> ',' | float_ [push_back(phoenix::ref(k),_1)])
					 *(float_ [push_back(phoenix::ref(k),_1)])
					)
					,
					space);

			return r;
		}



	template <typename Iterator>
		bool parse_k_vals(Iterator first, Iterator last, std::vector<float>& w)
		{			
			bool r = phrase_parse(first, last,

					//  grammar for k constants
					(
					 ("k(" >> float_[push_back(phoenix::ref(w), _1)] >> ')')
					)
					,
					//skip
					space | '*' | float_ >> '*' | '-'>>float_ >> '*'| '+' >> float_ >> '*'| "y(" >> float_ >> ')');

			return r;
		}


	template <typename Iterator>
		bool parse_constants(Iterator first, Iterator last, std::vector<float>& v)
		{
			bool r = phrase_parse(first, last,

					//  grammar for numerical constants & sign
					(
					 (float_[push_back(phoenix::ref(v), _1)] >> '*')
					)
					,
					//skip
					space);
			return r;
		}
	template <typename Iterator>
		bool parse_y_vals(Iterator first, Iterator last, std::map<float,float>& x)
		{
			float base =1.0;
			float exp = 0.0;
			bool r = phrase_parse(first, last,

					//  grammar for y variables
					//  this really looks like hell
					(
					 *("y(" >> float_[phoenix::ref(base)=_1] >> ")^(" >> float_[phoenix::ref(exp)=_1] >> '/' >> float_[phoenix::ref(exp)/=_1]\
						 [boost::bind(&update,boost::ref(x),boost::ref(base),boost::ref(exp))] >> ')' |
						 "y(" >> float_[phoenix::ref(base)=_1, boost::bind(&update,boost::ref(x),boost::ref(base),1.0f)] >> ')')
					)
					,
					//skip
					space | '*' | float_ >> '*' | '-'>>float_ >> '*'| '+' >> float_ >> '*'| "k(" >> float_ >> ')');


			return r;
		}


}

#endif

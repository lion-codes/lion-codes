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

#include <iostream>
#include <stdexcept>
#include <exception>
#include <mpi.h>

using namespace std;

#include "FQHE.h"

int main(int argc, char *argv[] ){

	try {
		FQHE *fqhe = new FQHE(&argc,&argv);
		delete fqhe;
	}
	catch(exception& e) {
		cerr << "ERROR \t" << e.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

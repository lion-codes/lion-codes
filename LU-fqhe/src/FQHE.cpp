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
#include <stdexcept>
#include <cstring>
#include <mpi.h>
#include <omp.h>

#include "FQHE.h"
#include "Node.h"
#include "RunInfo.h"
#include "PermHandler.h"
#include "MonteCarloLoop.h"

using namespace std;
using namespace MPI;


//// Default Constructor
// Creates a program object
// NULL objects
FQHE::FQHE() {
	node = NULL;
	runInfo = NULL;
	mcLoop = NULL;
	pHandler = NULL;
}

//// Constructor
// Creates a program object
// Parses the command line
FQHE::FQHE(int *argc,char ***argv) {

	// Initializes the MPI Environment
	MPI::Init(*argc,*argv);
	COMM_WORLD.Set_errhandler(ERRORS_THROW_EXCEPTIONS);

	// Declare variables
	int myRank;
	char *pFile;
	int nMC,nPerm,nMperBl,matSide;
	int mode;

	// Get the rank
	try {
		myRank = COMM_WORLD.Get_rank();
	}
	catch (MPI::Exception& e) {
		cerr << "ERROR \t " << e.Get_error_code() << " - " << e.Get_error_string()	<< endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}

	// Parse the command line
	try {
		commandLineParse(argc,argv,myRank,&nMC,&nPerm,&nMperBl,&matSide,&pFile,&mode);
	}
	catch(exception& e) {
		cerr << "ERROR \t Could not parse the command line " << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}

	////////////////////////////////////
	//// Discover the "Environment" ////
	////////////////////////////////////
	try {
		COMM_WORLD.Barrier();	
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << "\t Discovering environment" << endl;
#endif

		node = new Node();
	}
	catch(exception& e) {
		cerr << "ERROR \t Failed to detect the environment on node " << myRank << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}

	/////////////////////////////////
	//// Information for running ////
	/////////////////////////////////
	// Set up the information obtained from the command line (number of matrices, side of a matrix), and its derived values
	try {
		COMM_WORLD.Barrier();	
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << "\t Finding information to perform the decomposition" << endl;
#endif

		runInfo = new RunInfo(nMperBl,matSide,nPerm,node);
	}
	catch(exception& e) {
		cerr << "ERROR \t Failed to read the running information " << myRank << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}

	///////////////////////////////
	//// Read the permutations ////
	///////////////////////////////
	try {
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << "\t Reading permutations" << endl;
#endif
		pHandler = new PermHandler(myRank,runInfo,node, pFile, mode);
	}
	catch (exception& e) {
		cerr << "ERROR \t Failed to read the permutations on" << myRank << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}

	/////////////////////////////////
	//// Allocate all the memory ////
	/////////////////////////////////
	// Allocate all the shared memory; create a holder for all the constants
	try {
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << "\t Initializing the Monte Carlo loops" << endl;
#endif
		mcLoop = new MonteCarloLoop(*runInfo,nMC);

	} catch(exception& e) {
		cerr << "ERROR \t Failed to allocate the create the Monte Carlo loop" << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}
	//// Monte Carlo Loop
	try {
		COMM_WORLD.Barrier();	
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << "\t Running the Monte Carlo loops" << endl;
#endif
		mcLoop->perform(node,runInfo,pHandler,mode);

	} catch(exception& e) {
		cerr << "ERROR \t Failed ton run the Monte Carlo Loop on Node " << myRank << endl;
		COMM_WORLD.Abort(-1);
		Finalize();
		throw exception();
	}
#ifdef __VERBOSE__
		cout << "---------------------------------------------------------------------" << endl;
		cout << "INFO \t Node " << myRank << "\t Done." << endl;
#endif

	delete node;
	node = NULL;
	delete runInfo;
	runInfo = NULL;
	delete mcLoop;
	mcLoop = NULL;
	delete pHandler;
	pHandler = NULL;


	Finalize();


}

FQHE::~FQHE() {
	if(node != NULL) {
		delete node;
		node = NULL;
	}
	if(runInfo != NULL) {
		delete runInfo;
		runInfo = NULL;
	}
	if(mcLoop != NULL) {
		delete mcLoop;
		mcLoop = NULL;
	}

	if(pHandler != NULL) {
		delete pHandler;
		pHandler = NULL;
	}
}

void FQHE::commandLineParse(int *argc, char ***argv, int rank,int *nMC, int *nPerm, int *nMperBl, int *matSide, char **pFile, int *mode) {
	if(*argc < 12) {
		if (rank == 0) {
			cout << "Usage is:" << endl;
			cout << "./fqhe.x -p <permutation file> -mc <number of Monte Carlo loops> -N <number of matrices> -Npb <number of matrices to be treated per GPU block> -s <matrix side> -mode <LU/perm>" << endl;
		}
		throw exception();
	}
	else {
		for(int i = 1; i<*argc ; i+=2) {
			if (	!strcmp( (*argv)[i],"-s" ) 	) {
				*matSide = atoi((*argv)[i+1]);
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Side of a matrix\t\t\t" << *matSide << endl;
#endif
			}
			else if (	!strcmp( (*argv)[i],"-Npb")	) {
				*nMperBl = atoi((*argv)[i+1]);
#ifdef __VERBOSE__
				if( rank == 0 ) 	cout << "INFO \t Number of matrices per GPU block\t" << *nMperBl << endl;
#endif
			}
			else if (	!strcmp( (*argv)[i],"-N")	) {
				*nPerm = atoi((*argv)[i+1]);
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Number of permutations / matrices\t" << *nPerm << endl;
#endif
			}
			else if (	!strcmp( (*argv)[i],"-mc") 	) {
				*nMC = atoi((*argv)[i+1]);
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Number of Monte Carlo iterations\t" << *nMC << endl;
#endif
			}
			else if (	!strcmp( (*argv)[i],"-f") 	) {
				*pFile = (*argv)[i+1];
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Permutations/matrices file path\t" << *pFile << endl;
#endif
			}
			else if (	!strcmp( (*argv)[i],"-mode")) {
				if( !strcmp( (*argv)[i+1],"perm" )) {
					*mode = PERM;
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Computation mode \t\t\t" << *mode << " (Permutations)" << endl;
#endif
				} else {
					*mode = LU;
#ifdef __VERBOSE__
				if( rank == 0 )	cout << "INFO \t Computation mode \t\t\t" << *mode << " (LU Only)" << endl;
#endif
				}
			}
			else {
				if (rank == 0) {
					cout << "ERROR \t Invalid argument" << endl;
					cout << "Usage is:" << endl;
					cout << "./fqhe.x -f <permutation/matrix file> -mc <number of Monte Carlo loops> -N <number of permutations/matrices> -Npb <number of matrices to be treated per GPU block> -s <matrix side> -mode <LU/perm>" << endl;
				}
				throw exception();
			}
		}
	}


}

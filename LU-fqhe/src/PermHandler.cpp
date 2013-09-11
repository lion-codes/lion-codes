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
#include <mpi.h>
#include <unistd.h>
#include <omp.h>

#include "RunInfo.h"
#include "Node.h"
#include "GlobalInfo.h"
#include "PermHandler.h"

using namespace std;
using namespace MPI;

PermHandler::PermHandler() {
	localPerm = NULL;
	localMatrices = NULL;
}

PermHandler::PermHandler(int myRank, RunInfo *rInfo, Node *node, const char *filePath, const int mode) {

	// Read from RunInfo and Node the needed data
	int nElem 		= rInfo->getNElem();
	int nNodes 		= node->getGlobalInfo()->getNodCnt();
	int loadSize 		= rInfo->getLoadSize();
	int matSide		= rInfo->getMatSide();

	// Declare offsets and read size
	int startIdx	= 0;
	int endIdx 		= 0;
	int myIndices 	= 0;

	int readBytes 	= 0;
	int offsetBytes	= 0;
	int offset		= 0;

	File MPIFile;

	if( mode == PERM) {
#ifdef __VERBOSE__
	cout << "INFO \t Node " << myRank << " is attempting to load permutations" << endl;
#endif

		// Allocate the memory
		try {
			localPerm = new short[myIndices * loadSize];
			localMatrices = NULL;
		} catch (bad_alloc& e) {
			cerr << "ERROR \t Node " << myRank << " failed to allocate memory of size " << readBytes << " bytes" << endl;
			throw exception();
		}

		// Open the file, read, and close
		try {
			// Define offset and read size
			startIdx	= (int) ((float) (myRank * nElem) / nNodes);
			endIdx 		= (int) ((float) ((myRank+1) * nElem) / nNodes)-1;
			myIndices 	= endIdx-startIdx+1;

			readBytes 	= myIndices * loadSize * sizeof(short);
			offsetBytes	= startIdx * loadSize * sizeof(short);
			offset		= startIdx * loadSize;

			File MPIpermFile = File::Open(COMM_WORLD,filePath,MPI_MODE_RDONLY,MPI_INFO_NULL);
			MPIpermFile.Read_at_all(offset,localPerm,myIndices,MPI_SHORT);
			MPIpermFile.Close();
		}
		catch (MPI::Exception& e) {
			cerr << "ERROR \t Node " << myRank << " failed to read the file " << filePath << endl;
					throw exception();
		}
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << " successfully loaded its permutations, of size " << myIndices << " ( " << readBytes << " bytes )"<< endl;
#endif
	}
	else if ( mode == LU) {
#ifdef __VERBOSE__
	cout << "INFO \t Node " << myRank << "\t is attempting to load matrices" << endl;
#endif
		// Allocate the memory
		try {
			int nArrayElem = nElem * matSide * matSide * 2; // The total number of elements in the matrix file; they are complex numbers, hence the 2*

			startIdx	= (int) ((float) (myRank * nArrayElem) / nNodes);
			endIdx 		= (int) ((float) ((myRank+1) * nArrayElem) / nNodes)-1;
			myIndices 	= endIdx-startIdx+1;			// Total number of elements to read

			readBytes 	= myIndices * sizeof(float);	// Total size in bytes to read
			offsetBytes	= startIdx  * sizeof(float);	// Total offset in bytes
			offset		= startIdx;						// Total offset in elements


			localPerm = NULL;
#ifdef __DPREC__
			localMatrices = new double[myIndices];
#else
			localMatrices = new float[myIndices];
#endif
			cout << "INFO \t Node " << myRank << "\t startIdx = " << startIdx << " Endidx = " << endIdx << " Indices = " << myIndices << " Offset = " << offset << endl;


		} catch (bad_alloc& e) {
			cerr << "ERROR \t Node " << myRank << " failed to allocate memory of size " << readBytes << " bytes" << endl;
			throw exception();
		}
		// Open the file, read, and close
		try {

/*
			MPI::Info MPIInfo;
			MPIInfo.Create();
			MPIInfo.Set("coll_read_bufsize","4096");
*/			




//			MPIFile = File::Open(COMM_WORLD,filePath,MPI_MODE_RDONLY,MPIInfo);

			MPIFile = File::Open(COMM_WORLD,filePath,MPI::MODE_RDONLY,MPI::INFO_NULL);
			MPIFile.Set_view(offsetBytes,MPI_FLOAT,MPI_FLOAT,"native",MPI::INFO_NULL);
		}	
		catch (MPI::Exception& e) {
			cerr << "ERROR \t Node " << myRank << " failed to open the file " << filePath << endl;
			throw exception();
		}	
		try {
			double begin,end;

			begin = omp_get_wtime();

#ifdef __DPREC__
			MPIFile.Read_at_all(offsetBytes,localMatrices,myIndices,MPI_DOUBLE);
#else
		//	MPIFile.Read_at_all(offsetBytes,localMatrices,myIndices,MPI_FLOAT);
			MPIFile.Read(localMatrices,myIndices,MPI_FLOAT);
#endif
			end = omp_get_wtime();
			cout << "INFO \t Node " << myRank << "\t Time to read " << end-begin << endl;
		}
		catch (MPI::Exception& e) {
			cerr << "ERROR \t Node " << myRank << " failed to read the file " << filePath << endl;
			throw exception();
		}
		try {
			MPIFile.Close();
		}
		catch (MPI::Exception& e) {
			cerr << "ERROR \t Node " << myRank << " failed to close the file " << filePath << endl;
			throw exception();
		}
#ifdef __VERBOSE__
		cout << "INFO \t Node " << myRank << " successfully loaded its matrices, of size " << myIndices<< " ( " << readBytes << " bytes )"<< endl;
#endif
	}
}

// Return the address of to which localPerm is pointing to
short* PermHandler::getLocalPerm() {
	return localPerm;
}

#ifdef __DPREC__
double* PermHandler::getLocalMatrices() {
	return localMatrices;
#else
float* PermHandler::getLocalMatrices() {
	return localMatrices;
}
#endif

// Destructor
PermHandler::~PermHandler() {
	if(localPerm != NULL) {
		delete [] localPerm;
		localPerm = NULL;
	}
	if(localMatrices != NULL) {
		delete [] localMatrices;
		localMatrices = NULL;
	}
}

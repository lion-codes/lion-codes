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

void lanczos(complex double * A, 	// chunk of A
		complex double * evecs, //the eigenvectors
		double * evals,		//evals, real
		int n, 			// full size of A
		int m,			// rows of A for this process
		int myOffset,			// where to begin			
		int subSize,			// the subspace size
		int commSize,			// MPI size
		int commRank);			// MPI rank


// a harness for testing lanczos

int main(int argc, char * argv[] ){


	MPI_Init(&argc, &argv);

	int commSize, commRank;
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commRank);


	if (argc < 4){
		fprintf(stderr,"USAGE : lan.x <subspace_size> <matrix_side> <matrix file (double complex)>\n");
		MPI_Finalize();
		exit(1);
	}

	// matrix size
	int matSize 		= atoi(argv[2]);

	if ((matSize > _MAX_SIZE) || (matSize < 0)){
		fprintf(stderr,"matrix exceeds bounds\n");
		MPI_Finalize();
		exit(1);
	}

	//make a test matrix

	// subspace size
	int subSize 	= atoi(argv[1]);
	int begRow     	= (int) ((float) (commRank * matSize) / commSize);
	int endRow     	= (int) ((float) ((commRank+1) * matSize) / commSize)-1;
	int myRows     	= endRow-begRow+1;
	int myOffset	= begRow;


	complex double * A = (complex double *) malloc(sizeof(complex double) * myRows * matSize);

	MPI_File in;

	int ierr = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
	if (ierr) {
		if (commRank == 0) fprintf(stderr, "%s: Couldn't open file %s\n", argv[0], argv[3]);
		MPI_Finalize();
		exit(1);
	}

	MPI_File_read_at_all(in, myOffset*matSize*2*sizeof(MPI_DOUBLE), A, myRows*matSize*2, MPI_DOUBLE, MPI_STATUS_IGNORE);


	MPI_File_close(&in);
#if 0
#ifdef _DEBUG_LANCZOS
	// dump out


		fprintf(stderr,"\n");

	for (int i=0; i<myRows; i++){

		for (int j=0; j<matSize; j++)
			fprintf(stderr,"%i,%i,%i %f+%fi ",commRank,i+myOffset, j,creal(A[i+j*myRows]),cimag(A[i+j*myRows]));

		fprintf(stderr,"\n");
	}

		fprintf(stderr,"\n");

	fprintf(stderr,"myOffset : %i matSize : %i myRows : %i commRank : %i\n",myOffset,matSize,myRows,commRank);
#endif
#endif
	
	complex double * evecs	= (complex double*) malloc(sizeof(complex double) * matSize * subSize);
	double * evals	= (double*) malloc(sizeof(double) * subSize);

	lanczos(A, evecs, evals, matSize, myRows, myOffset, subSize, commSize, commRank);

	free(A); free(evecs); free(evals);
	MPI_Finalize();


	return 0;

}

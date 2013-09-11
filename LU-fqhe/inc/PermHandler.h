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

#ifndef PERMHANDLER_H_
#define PERMHANDLER_H_

enum MODE { LU = 0, PERM = 1};

class PermHandler {
	private:
		short *localPerm; // All the permutations local to a node

#ifdef __DPREC__
		double *localMatrices; // All of the matrices local to a node
#else
		float *localMatrices; // All of the matrices local to a node
#endif

	public:
		PermHandler();
		~PermHandler();
		PermHandler(int myRank, RunInfo *rInfo, Node *node, const char *filePath, const int mode);

		short* getLocalPerm(); // Get the array of permutations

#ifdef __DPREC__
		double* getLocalMatrices();
#else
		float* getLocalMatrices();
#endif

};


#endif /* PERMHANDLER_H_ */

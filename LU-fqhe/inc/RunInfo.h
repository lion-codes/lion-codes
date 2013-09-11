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

#ifndef RUNINFO_H_
#define RUNINFO_H_

#include "Node.h"

class RunInfo {
	private:
		int nMperBl; // Number of matrices treated per block
		int matSide; // Matrix side
		int loadSize; // Load size
		int nElem;		// Total number of permutations / matrices
		int nElemPNode; // Total number of permutations / matrices _per_node_

		int scrSpace;// Scratch space; power of 2 above matSide
		int scrSpace2; // Power of 2 below matSide

		int warpPerMat; // Number of warps per matrix

		int maxBlocks; // Max number of blocks for 1 kernel

	public:
		//// Constructors
		RunInfo();
		RunInfo(int nMperBl, int matSide, int nPerm, Node *node);

		//// Getters
		int getNMperBl();
		int getMatSide();
		int getLoadSize();

		int getScrSpace();
		int getScrSpace2();

		int getWarpPerMat();
		int getMaxBlocks();

		int getNElem();
		int getNElemPnode();

		static inline int p2Above (int N);
};

#endif /* RUNINFO_H_ */

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
#include <cstdlib>
#include <iostream>

#include "RunInfo.h"
#include "Node.h"
#include "LocalInfo.h"
#include "GlobalInfo.h"
#include "GpuInfo.h"
#include "SingleGpuInfo.h"

//// Constructors
// Default
RunInfo::RunInfo() {
	nMperBl		= 0; // Number of matrices treated per block
	matSide 	= 0; // Matrix side
	loadSize 	= 0; // Load size
	nElem		= 0; // Total number of permutations / matrices
	nElemPNode	= 0; // Total number of permutations / matrices _per_node_

	scrSpace 	= 0; // Scratch space; power of 2 above matSide
	scrSpace2 	= 0; // Power of 2 below matSide

	warpPerMat 	= 0; // Number of warps per matrix

	maxBlocks 	= 0; // Max number of blocks for 1 kernel
}


RunInfo::RunInfo(int nMperBl, int matSide, int nElem, Node *node) {
	this->nMperBl 	= nMperBl;
	this->matSide 	= matSide;
	this->nElem	= nElem;

	int nNodes	= node->getGlobalInfo()->getNodCnt();

	nElemPNode 	= nElem/nNodes;

	scrSpace 	= RunInfo::p2Above(matSide);
//	scrSpace2 	= scrSpace/2;
	scrSpace2	= 16;

	int nGPU = node->getLocalInfo()->getGpuInfo()->getGpuCnt();

	if(nGPU > 0) {
		int nWarp 		= node->getLocalInfo()->getGpuInfo()->getSingleGpuInfo(0).getWarpSize();
		maxBlocks 		= node->getLocalInfo()->getGpuInfo()->getSingleGpuInfo(0).getMaxGridSize();
		warpPerMat 		= matSide*matSide/nWarp+1;
	}
	else {
		warpPerMat = 0;
		maxBlocks = 0;
	}


#ifdef __PADDED__
	loadSize = scrSpace;
#else
	loadSize = matSide;
#endif	

}

//// Getters
int RunInfo::getNMperBl() {
	return nMperBl;
}

int RunInfo::getMatSide() {
	return matSide;
}

int RunInfo::getLoadSize() {
	return loadSize;
}

int RunInfo::getNElem() {
	return nElem;
}

int RunInfo::getNElemPnode() {
	return nElemPNode;
}

int RunInfo::getScrSpace() {
	return scrSpace;
}

int RunInfo::getScrSpace2() {
	return scrSpace2;
}

int RunInfo::getMaxBlocks() {
	return maxBlocks;
}

int RunInfo::getWarpPerMat() {
	return warpPerMat;
}


// Round up to next higher power of 2 (return x if it's already a power of 2).
inline int RunInfo::p2Above (int N) {
    if (N < 0)
        return 0;
    --N;
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    return N+1;
}

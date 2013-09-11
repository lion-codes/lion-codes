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

#include "SingleGpuInfo.h"

SingleGpuInfo::SingleGpuInfo() {
	idx	= -1;
	warpSize = 0;
	maxThreadsPerBlock = 0;
	maxThreadsDim = 0;
	maxGridSize = 0;
	totalGlobalMem = 0;
}

SingleGpuInfo::SingleGpuInfo(int idx, int warpSize, int maxThreadsPerBlock, int maxThreadsDim, int maxGridSize,size_t totalGlobalMem) {
	this->idx	= idx;
	this->warpSize = warpSize;
	this->maxThreadsPerBlock = maxThreadsPerBlock;
	this->maxThreadsDim = maxThreadsDim;
	this->maxGridSize = maxGridSize;
	this->totalGlobalMem = totalGlobalMem;
}

int SingleGpuInfo::getWarpSize() {
	return warpSize;
}

int SingleGpuInfo::getMaxGridSize() {
	return maxGridSize;
}

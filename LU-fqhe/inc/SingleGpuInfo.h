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

#ifndef SINGLEGPUINFO_H_
#define SINGLEGPUINFO_H_

#include <cstdio>
#include <cstdlib>

class SingleGpuInfo {
	private:
		int 	idx;
		int    	warpSize;                   /**< Warp size in threads */
		int    	maxThreadsPerBlock;         /**< Maximum number of threads per block */
		int    	maxThreadsDim;           	/**< Maximum size of each dimension of a block */
		int    	maxGridSize;             	/**< Maximum size of each dimension of a grid */
		size_t 	totalGlobalMem;             /**< Global memory available on device in bytes */

	public:
		SingleGpuInfo();
		SingleGpuInfo(int idx, int warpSize, int maxThreadsPerBlock, int maxThreadsDim, int maxGridSize,size_t totalGlobalMem);

		int getWarpSize();
		int getMaxGridSize();

/*
		void setIdx(int idx);
		void setWarpSize(int warpSize);
        void setMaxThreadsPerBlock(int maxThreadsPerBlock);
        void setMaxThreadsDim(int maxThreadsDim);
        void setMaxGridSize(int maxGridSize);
        void setTotalGlobalMemSize(size_t totalGlobalMem);*/
};


#endif /* SINGLEGPUINFO_H_ */

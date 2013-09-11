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

#ifndef GPUINFO_H_
#define GPUINFO_H_

#include <cuda.h>
#include <vector>

#include "EnvInfo.h"
#include "SingleGpuInfo.h"

using namespace std;

class GpuInfo : public EnvInfo {
	private:
		int gpuCnt;							// Number of GPUs
		vector<SingleGpuInfo> gpuProp;		// Characteristics of each of them: warpSize, maxThPBlock, maxThDim, maxGridSize, totalglobalmem

	public:
		GpuInfo();
		~GpuInfo();

		int getGpuCnt();
		SingleGpuInfo getSingleGpuInfo(int N);

};


#endif /* GPUINFO_H_ */

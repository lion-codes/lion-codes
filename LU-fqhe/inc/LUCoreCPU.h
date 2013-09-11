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

#ifndef LUCORECPU_H_
#define LUCORECPU_H_

#include <cuda.h>

#include "Node.h"
#include "RunInfo.h"

namespace LUCoreCPU {
		void CALL(float2 *matrices,float2 *result, int num, RunInfo *rInfo, int rank, int cpuCnt);
		void determinant(float2 * inputMatrix,float2 *det, int matSide, int loadSize);
}

#endif /* LUCORECPU_H_ */

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

#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <cuda.h>

void checkCUDAError(const char *msg,const char *file,const int line); 

__global__ void create_ptr(float2 **q_A,float2 **q_B,float2 **q_C,float2 *q_temp,float2 *q_tempB,float2 *q_complete) ;
__global__ void copy_to_temp(float2 *q_temp, float2 *q_complete) ;
__global__ void printf_ptr(float2 **q_cblas,float2 *q); 

__device__ float cabs_sq(float2 input);

#endif

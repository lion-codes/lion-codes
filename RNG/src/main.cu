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


// A simple RNG based around the unpredictability of the scheduler
// Tested on a single M2070 by WJB/PYT
// FYI [0,1] distribution is heavily skewed toward bounds
//


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void rng_cuda(float * out){

	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float values[512];

	values[threadIdx.x] = 0;
	values[threadIdx.x] += clock();
	for (int i=0; i<16; ++i) values[threadIdx.x] += values[(threadIdx.x+32+i)%512];
	for (int i=0; i<16; ++i) values[(threadIdx.x+32+i)%512] += values[threadIdx.x];

	out[t_idx] = (cos(values[threadIdx.x])+1.0) / 2.0;
	//out[t_idx] = ((float) ((int) values[threadIdx.x] % 100)) / 100.0;


}



int main(int argc, char * argv[]){

	dim3 threads=512, blocks=10;
	int size = threads.x*blocks.x*sizeof(float);

	float * d_out, *out;
	cudaMalloc((void**) &d_out, size);
	out = (float*) malloc(size);


	rng_cuda<<<blocks,threads>>>(d_out);
	rng_cuda<<<blocks,threads>>>(d_out);
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
	cudaFree(d_out);

	for (int i=0; i<size / sizeof(float); i++) printf("%f \n",out[i]);

	return 0;
}

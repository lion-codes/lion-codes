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


#ifndef FLOAT2OPS_H_
#define FLOAT2OPS_H_

#include <cuda.h>
#include <cmath>

#ifdef __DPREC__
__inline__ __device__ double2 operator+(double2 inputA, double2 inputB){

	double2 ret;
	ret.x=inputA.x+inputB.x;
	ret.y=inputA.y+inputB.y;

	return ret;
}
__inline__ __device__ double2 operator-(double2 inputA, double2 inputB){

	double2 ret;
	ret.x=inputA.x-inputB.x;
	ret.y=inputA.y-inputB.y;
	return ret;

}
__inline__ __device__ __host__ double2 operator*(double2 inputA, double2 inputB){

	double2 ret;
	ret.x = inputA.x*inputB.x-inputA.y*inputB.y;
	ret.y = inputA.y*inputB.x+inputA.x*inputB.y;
	return ret;

}
__inline__ __device__ double2 operator*(double inputA, double2 inputB){

	double2 ret;
	ret.x = inputA*inputB.x;
	ret.y = inputA*inputB.y;
	return ret;

}
__inline__ __device__ __host__ double2 operator/(double2 inputA, double2 inputB){

	double2 ret;
	float den = inputB.x*inputB.x+inputB.y*inputB.y;

	ret.x = (inputA.x*inputB.x+inputA.y*inputB.y)/den;
	ret.y = (inputA.y*inputB.x-inputA.x*inputB.y)/den;
	return ret;

}
__inline__ __device__ double2 operator/(double inputA, double2 inputB){

	double2 ret;
	float den = inputB.x*inputB.x+inputB.y*inputB.y;
	ret.x = inputA*inputB.x/den;
	ret.y = -inputA*inputB.y/den;
	return ret;

}
__inline__ __host__ __device__ double2 powc(double n, double rad, double ang){

	double2 ret;
	double tmp = pow(rad,n);
	double tmp2 = cos(n*ang);
	ret.x = tmp * tmp2;
	ret.y = tmp * sqrt(1-tmp2*tmp2);
	return ret;
}
#else

// Addition of float2
__inline__ __device__ float2 operator+(float2 inputA, float2 inputB){

	float2 ret;
	ret.x=inputA.x+inputB.x;
	ret.y=inputA.y+inputB.y;

	return ret;
}

// Substraction of float2
__inline__ __device__ float2 operator-(float2 inputA, float2 inputB){

	float2 ret;
	ret.x=inputA.x-inputB.x;
	ret.y=inputA.y-inputB.y;
	return ret;

}

// Multiplication of float2
__inline__ __device__ __host__ float2 operator*(float2 inputA, float2 inputB){

	float2 ret;
	ret.x = inputA.x*inputB.x-inputA.y*inputB.y;
	ret.y = inputA.y*inputB.x+inputA.x*inputB.y;
	return ret;

}
// Multiplication by a scalar value
__inline__ __device__ float2 operator*(float inputA, float2 inputB){

	float2 ret;
	ret.x = inputA*inputB.x;
	ret.y = inputA*inputB.y;
	return ret;
}

// Division of float2
__inline__ __device__ __host__ float2 operator/(float2 inputA, float2 inputB){

	float2 ret;
	float den = inputB.x*inputB.x+inputB.y*inputB.y;

	ret.x = (inputA.x*inputB.x+inputA.y*inputB.y)/den;
	ret.y = (inputA.y*inputB.x-inputA.x*inputB.y)/den;
	return ret;
}

// Division of float2
__inline__ __device__ float2 operator/(float inputA, float2 inputB){

	float2 ret;
	float den = inputB.x*inputB.x+inputB.y*inputB.y;
	ret.x = inputA*inputB.x/den;
	ret.y = -inputA*inputB.y/den;
	return ret;
}

// 
__inline__ __host__ __device__ float2 powc(float n, float rad, float ang){

	float2 ret;
	float tmp = powf(rad,n);
	float tmp2 = cos(n*ang);
	ret.x = tmp * tmp2;
	ret.y = tmp * sqrt(1-tmp2*tmp2);
	return ret;
}

/*
// Equality
__inline__ __host__ __device__ void operator=(float2 inputA,float2 inputB) {
	inputA.x = inputB.x;
	inputA.y = inputB.y;
}	
*/



#endif

#endif /* FLOAT2OPS_H_ */

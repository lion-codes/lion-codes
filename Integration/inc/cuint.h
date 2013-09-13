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

#ifndef _CUINT_H_
#define _CUINT_H_

namespace cuint {
	
	void STRAPZ(float *V,float *out,float dx, int N); 
	void DTRAPZ(double *V,double *out,double dx, int N); 

/*	template <typename T>
	void TRPZ(T *V, T *X,T *out, int N) {
		//// Cast thrust pointer
		thrust::device_ptr<T> th_ptr = thrust::device_pointer_cast(V);
		thrust::device_vector<T> th_V (th_ptr,th_ptr + N);

		th_ptr = thrust::device_pointer_cast(X);
		thrust::device_vector<T> th_X (th_ptr,th_ptr + N);

		thrust::device_vector<T> tmp(th_V.size()-1);

		//// Apply transform to calculate f[xi+1]*[xi]
		thrust::transform(	th_V.begin(),
					th_V.end()-1,
					th_X.begin()+1,
					tmp.begin(),
					thrust::multiplies<T>());

		//// Calculate the corresponding sum
		*out = thrust::reduce(	tmp.begin(),
					tmp.end() - 1,
					(th_V[N-1]*th_X[N-1]-th_V[0]*th_X[0]),
					thrust::plus<T>() );
		
		//// Apply transform to calculate -f[xi]*[xi-1]
		thrust::transform(	th_V.begin() + 1,
					th_V.end(),
					th_X.begin(),
					tmp.begin(),
					thrust::multiplies<T>());
		
		//// Calculate corresponding sum
		*out = thrust::reduce(	tmp.begin(),
					tmp.end()-1,
					*out,
					thrust::negate<T>());

		*out *= (T)0.5;
	}
*/	



	// Trapezoidal integrator
/*	template <typename T>
	void TRPZ(T *V, T *out, T dx, int N); 

	template <typename T>
	void TRPZ(T *V, T *X, T *out, int N); 
*/
/*	// Simpson integrator
	template < typename T >
	SIMPS(T *V, T *out, T dx, int N); 

	template < typename T >
	SIMPS(T *V, T *X, T *out, int N); 
*/
}
	

#endif


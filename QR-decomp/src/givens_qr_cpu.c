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

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "main.h"

void givens_qr_cpu(float *mats, int size, float *q) {
	float2 *q_complete, *matrices;


	// Create a temporary array for storage
	matrices = (float2*)malloc(sizeof(float2) * size * MATRIX_SIDE * MATRIX_SIDE);
	q_complete = (float2*)malloc(sizeof(float2) * size * MATRIX_SIDE * MATRIX_SIDE);
	

	for (int k=0; k<size; k++) {
		for (int i=0; i<MATRIX_SIDE; i++) {		
			for (int j=0; j<MATRIX_SIDE; j++) {
				q_complete[j+i*MATRIX_SIDE+k*MATRIX_SIDE*MATRIX_SIDE].x = (i==j) ? 1.0 :0.0;		
				q_complete[j+i*MATRIX_SIDE+k*MATRIX_SIDE*MATRIX_SIDE].y = 0.0;		
			}
		}
	}	
		
	int qsize = size * MATRIX_SIDE * MATRIX_SIDE;	
	

	// Copy data
	for(int k=0;k<size;k++) {
		for(int i = 0;i<MATRIX_SIDE;i++) {
			for(int j = 0;j<MATRIX_SIDE;j++) {
				matrices[ j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].x = mats[(2*j+i*MATRIX_SIDE*2)+k*MATRIX_SIDE*MATRIX_SIDE*2]; 
				matrices[ j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE].y = mats[(2*j+i*MATRIX_SIDE*2+1)+k*MATRIX_SIDE*MATRIX_SIDE*2]; 
			}
		}
	}	

	
	double begin,end;

	float2 u,v,c,s,up,lo;
	float f,g,den;

//	fprintf(stderr,"Starting OMP routine...\n");
	begin = omp_get_wtime();
	// For all matrices
	#pragma omp parallel for private(u,v,c,s,f,g,den,up,lo)
	for(int k = 0;k<size;k++) {
		int mymatrix = k*MATRIX_SIDE*MATRIX_SIDE;
		// For all columns
		for(int j = 0;j<MATRIX_SIDE-1;j++) {
			// For all rows
			for(int i=MATRIX_SIDE-1;i>=j+1;i--) {
				// Get a and b	
				u = matrices[ j + (i-1)*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];
				v = matrices[ j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];

				// Calculate c and s
				f = u.x*u.x + u.y*u.y;
				g = v.x*v.x + v.y*v.y;

				if( g < 2e-16 ) {
					c.x = 1.0f;
					c.y = 0.0f;

					s.x = 0.0f;
					s.y = 0.0f;

				} else if (f< 2e-16) {
					c.x = 0.0f;
					c.y = 0.0f;

					// s = conj(v)/g
					den = 1.0f/g;
					s.x = v.x*den; 
					s.y = -v.y*den;
				} else {
					// r = sqrt(f + g)
					den = 1.0f/sqrt(f + g);
					// c = f/r
					c.x = sqrt(f)*den;
					c.y = 0.0f;

					// s = x/f * conj(y) / r
					// den = -1/(f*r)
					den  = den / sqrt(f);
					
					s.x = (u.x*v.x + u.y*v.y)*den;
					s.y = (u.y*v.x - u.x*v.y)*den;
				}	

				// Multiply the full rows 
				for(int l = 0; l < MATRIX_SIDE; l++) {
					u = matrices [ l + (i-1)*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];
					v = matrices [ l + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];

					// c*u + s*v
					// Perform product: real part 
					f = (u.x*c.x - u.y*c.y) + (v.x*s.x - v.y*s.y); 
					// Perform product: imaginary part
					g = (u.x*c.y + u.y*c.x) + (v.x*s.y + v.y*s.x);

					up.x = f;
					up.y = g;

					// -conj(s)*c + c*v
					// Perform product: real part 
					f = -(u.x*s.x + u.y*s.y) + (v.x*c.x - v.y*c.y); 
					// Perform product: imaginary part
					g = (u.x*s.y - u.y*s.x) + (v.x*c.y + v.y*c.x);
					
					lo.x = f;
					lo.y = g;
					
					// Write back in global
					matrices [ l + (i-1)*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ] = up;
					matrices [ l + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ] = lo;

			
					//// Accumulate in Q
					u = q_complete[ l + (i-1)*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];
					v = q_complete[ l + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE];
					// c*u + s*v
					// Perform product: real part 
					f = (u.x*c.x - u.y*c.y) + (v.x*s.x - v.y*s.y); 
					// Perform product: imaginary part
					g = (u.x*c.y + u.y*c.x) + (v.x*s.y + v.y*s.x);

					up.x = f;
					up.y = g;

					// -conj(s)*c + c*v
					// Perform product: real part 
					f = -(u.x*s.x + u.y*s.y) + (v.x*c.x - v.y*c.y); 
					// Perform product: imaginary part
					g = (u.x*s.y - u.y*s.x) + (v.x*c.y + v.y*c.x);

					lo.x = f;
					lo.y = g;
					
					// Write back in global
					q_complete[ l + (i-1)*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ] = up;
					q_complete[ l + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ] = lo;
				}	

				// Introduce exact zero
				matrices [ j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ].x = 0.0f;
				matrices [ j + i*MATRIX_SIDE + k*MATRIX_SIDE*MATRIX_SIDE ].y = 0.0f;
			}
		}
	}	
	end = omp_get_wtime();
	printf("CPU  \t %f\n",end-begin);


	#ifdef _QR_VERBOSE_
		for (int k=0; k<size; k++){
			printf("q %i\n",k);
			for (int i=0; i<MATRIX_SIDE; i++) {
				for (int j=0; j<MATRIX_SIDE; j++){
					printf("%f+%fi, ",q_complete[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].x,
							q_complete[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].y);
			
				}	
					printf(";\n");
			}	

		}
		for (int k=0; k<size; k++){
			printf("r_mat %i\n",k);
			for (int i=0; i<MATRIX_SIDE; i++){
				for (int j=0; j<MATRIX_SIDE; j++)
					printf("%f+%fi, ",matrices[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].x,
							matrices[j+MATRIX_SIDE*i+k*MATRIX_SIDE*MATRIX_SIDE].y);


				printf(";\n");
			}

		}


	#endif



	free(matrices);
	free(q_complete);
}						
				
					







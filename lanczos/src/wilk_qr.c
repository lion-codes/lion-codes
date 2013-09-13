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

#include "main.h"


// get sign
// definition from Bindel et al for Givens rotations

complex double sgn(complex double val) { 
	return (creal(val) >0 || cimag(val)>0) ? val / fabs(val) : 1.0+0.0*_Complex_I ;
}

// get the largest magnitude component
// definition from Bindel et al for Givens rotations

double magc(complex double val){ 

	double rd = fabs(creal(val));
	double id = fabs(cimag(val));

	return (rd > id) ? rd : id;
}

// get the complex conjugate

complex double conj(complex double val){ 
	return (creal(val) - cimag(val)*_Complex_I); 
}


// qr algorithm for triagonal matrices with wilkinson shifts

void wilk_qr(complex double * alpha, 	// diag entires	
		complex double * beta, 	// minor diags
		complex double * evecs, // eigenvectors
		int n){			// order

#ifdef _CALC_EVECS

	// multiplicative factors in gemm
	complex double mula 	= 1.0+0.0*_Complex_I;
	complex double mulb 	= 0.0+0.0*_Complex_I;

	// more args for gemv
	char type = 'N';
	int matSize=n,info,dim=matSize;

	// set up scratch space for making evectors
	complex double * scratch = (complex double * ) malloc(sizeof(complex double) * n*n);

	for (int i=0; i<n*n; i++){ 
		scratch[i]=0.0+0.0+_Complex_I;
	}

	for (int i=0; i<n; i++) scratch[i+n*i]=1.0+0.0+_Complex_I;

	// initialize evecs

	for (int i=0; i<n*n; i++){ 
		evecs[i]=scratch[i];
	}

#endif

	int m=n;
	int it=0;
	int maxIts = n*n*10;

	// Wilkinson shift
	complex double lam1,lam2; // Eigenvalues in the last 2x2 principal submatrix
	double d1,d2; // Distance from the eigenvalue to a[n][n] in the last 2x2 principal submatrix
	complex double mu;
	complex double d;

	// Convergence
	double last;

	// Inner QR
	complex double c,s,c2,s2; 		// Cos, sin, and square
	complex double x,y,r;	  	// Givens rotations
	complex double t,p;	  		// temporaries
	complex double ak,akp1,bk,bkp1,bkp2; 

	double f,g,th;


	while ( m > 1){
		// calculate wilkinson shift
		// ~ evalue for last 2x2 block
		d = (alpha[m-2]-alpha[m-1]) / (2.0+0.0*_Complex_I);

		lam1	= alpha[m-1] + d + sqrt(d*d+beta[m-2]*beta[m-2]);	
		lam2	= alpha[m-1] + d - sqrt(d*d+beta[m-2]*beta[m-2]);	
		
		// Calculate distance to alpha[m] 
		d1	= cabs(lam1);
		d2	= cabs(lam2); 
		mu = ( d1 < d2 ) ? lam1 : lam2;

		x = alpha[0]-mu;
		y = beta[0];

		last = (magc(alpha[m-2]) + magc(alpha[m-1]));

		for (int k=0; k<m-1; k++){


			if (m>2){
				// calculate givens rotation elements
				// From "On computing Givens rotations reliably and efficiently"
				// Bindel, Demmel, Kahan, Marques
				f	= cabs(x);
				g	= cabs(y);
				if( cabs(y) < 2e-16 )
				{
					c = 1;
					s = 0;
				}
				else if (cabs(x) < 2e-16 )
				{
					c = 0;
					s = conj(y)/g;
				}
				else
				{
					r = sqrt(f*f + g*g);
					c = f/r;
					s = -x/f * conj(y) / r;
				}	
			} else{
				// For the last iteration, we need to diagonalize the remaining 2x2 exactly
				th	= 0.5*atan2(2*beta[0], (alpha[1] - alpha[0]));
				s	= sin(th);
				c	= cos(th);
			}

			s2	= s*s;
			c2	= c*c;
			p	= s*c;
			t       = 2*p*beta[k];

			ak 	= c2*alpha[k]  + s2*alpha[k+1]-t;
			akp1 	= c2*alpha[k+1]+ s2*alpha[k]  +t;

			bkp1	= (c2-s2)*beta[k] + p*(alpha[k]-alpha[k+1]);


			if(k > 0) {
				beta[k-1] = c*x - s*y;
			}	

			alpha[k] = ak;
			alpha[k+1] = akp1;
			beta[k] = bkp1;

			x	= beta[k];

			if( k < m-2 ) { 
				y		= -s*beta[k+1]; 
				beta[k+1] 	*= c;
			}	



#ifdef _CALC_EVECS
			// build up the eigenvectors by products of Givens rotations
			// update scratch

			if (k>0){

				scratch[k-1+(k-1)*n]	= 1;
				scratch[k-1+k*n]	= 0;
				scratch[k+(k-1)*n]	= 0;
				scratch[k+k*n]		= 1;
			}

			scratch[k+k*n]		= c;
			scratch[k+(k+1)*n]	= s;
			scratch[k+1+k*n]	= -s;
			scratch[k+1+(k+1)*n]	= c;


			// do the work
			zgemm_( &type, 	&type, &n, &n, &n, &mula, evecs, &dim, scratch, &dim, &mulb, evecs, &dim);

#endif

		}

		for (int i=0; i<n; i++)
			printf("%f+%fi ",creal(alpha[i]),cimag(alpha[i]));
		printf("\n");


		// check convergence
		if( m > 1 ) {
			printf("%i %f \n",m,magc(beta[m-2]));

			if (magc(beta[m-2]) < 0.001*fabs(last- (magc(alpha[m-2]) + magc(alpha[m-1]))))
			m -= 1;
		}
		it++;

		if (it>maxIts){
			fprintf(stderr,"QR algorithm failed to converge; exceeded maximum iterations ~ N*N\n");
			MPI_Finalize();
			exit(1);
		}
	}
	free(scratch);
}


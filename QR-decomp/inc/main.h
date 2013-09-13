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

#ifndef _MAIN_H
#define _MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>

//#define NUM_MATRICES		1
//#define STREAMS			8
#define MATRIX_SIDE	32	
#define _MAX_SIZE 1024
#define NMPBL	32	
#define NTH 1024 

#define global_row_i	memoryStride + i*MATRIX_SIDE + vectorIndex
#define global_row_j	memoryStride + j*MATRIX_SIDE + vectorIndex
#define global_col_j	memoryStride + vectorIndex*MATRIX_SIDE + j

#define smemstride	MATRIX_SIDE*myMatrix

//#define TEST_ME_G
//#define _QR_VERBOSE_ 
//#define TEST_MAT

#define global_upper_row 	memoryStride + (MATRIX_SIDE-i-2) * MATRIX_SIDE + vectorIndex
#define global_lower_row 	memoryStride + (MATRIX_SIDE-i-1) * MATRIX_SIDE + vectorIndex
#define global_column 		memoryStride + vectorIndex * MATRIX_SIDE + col
#define global_row		memoryStride + i*MATRIX_SIDE + vectorIndex

#define rotation_ele_index	vectorIndex

#define sub_diag_mask		(vectorIndex == i-1)
#define for_diag_mask		(vectorIndex >= i)
#define above_diag_mask		(vectorIndex > i)
#define	last_col_mask		(vectorIndex == MATRIX_SIDE-1)
#define	last_two_col_mask	(vectorIndex >= MATRIX_SIDE-2)
#define	diag_mask		(vectorIndex == i)		



#endif

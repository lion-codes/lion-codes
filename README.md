lion-codes
==========

GPU research and teaching applications

Fsolve
------
The beginnings of a non-linear equation solver f(x)=0 based on Newton-Raphson, to assist in constructing chemical kinetic models of metabolic networks. The application initially parses a large set of equations, finds an analytic Jacobian, and then iteratively updates the 'best' solution x with delta by solving J.delta = -f, using GMRES from CUSP. Solve, parameter/Jacobian/function updates are all evaluated on GPU.

Contributor : A. Khodayari

Integration
-----------
Performs a simple composite trapezoidal or Simpsons 1D integration of a data set, approximately three times more performant than similar routines executed on a single CPU socket (recent Intel device). (TODO: Simpsons).

Lanczos
-------
A GPU version of the eigen-decomposition method for Hermitian matrices. Here, the matrix to be diagonalized is partitioned among GPUs, which essentially perform BLAS operations. The output tridiagonal matrix is diagonalized on the root MPI process using the tridiagonal QR algorithm, and eigenvectors may be constructed in the space of the original input (Hermitian) matrix. Designed for use with condensed matter applications. (TODO: sparse matrices, restart option to save space)

Contributor: F. Spiga

LU-FHQE
-------
This application uses a kernel for batch LUP decomposition, designed to operate in the range intermediate to the routines available in CUBLAS and MAGMA. In this particular application, LUP is used to find the determinant of many small matrices, working alongside a CPU version of Crout's approach. These determinants in turn contribute to the construction of wavefunctions, useful in Monte Carlo studies of the fractional quantum Hall effect. The symmetrization of FQHE wavefunctions scales as ~ n!

Contributor: Sreejith G. J.

QR-decomp
---------
Several GPU algorithms using variations on the Givens rotation based approach, with performance exceeding MKL in specific instances.


RNG
---
A cheap RNG based on the unpredictability of the scheduler; distribution is heavily skewed toward [0,1] bounds


WJB + PYT 09/13

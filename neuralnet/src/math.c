#include <cublasXt.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "math.h"

//Same as: pow(500, 3)
//500^3 was chosen empirically, it may need to be changed in other system
#define GPU_THRESHOLD 125000000

/*
 * This method represent the well known sigmoid activation function used very
 * often in artificial neural networks. The math resolves to:
 *
 * f(x) = 1 / (1 + e^(-x))
 *
 */
double math_sigmoid_fx(double x) {
	return 1.0 / (1 + exp(-x));
}

/*
 * This method represent the derivative of the sigmoid function.
 * The math resolves to:
 *
 * f(x) = g(x)*(1 - g(x))
 *
 * where
 * g(x): is the sigmoid function
 *
 */
double math_sigmoid_dx(double x) {
	return math_sigmoid_fx(x) * (1 - math_sigmoid_fx(x));
}

/*
 * This method represent the well known rectifier activation function used very
 * often in artificial neural networks. The math resolves to:
 *
 * f(x) = max(0, x)
 *
 */
double math_rectifier_fx(double x) {
	return fmax(0.0, x);
}

/*
 * This method represent the derivative of the rectifier function.
 * The math resolves to:
 *
 * f(x) = {0 if x <= 0; 1 if x > 0}
 *
 */
double math_rectifier_dx(double x) {
	return (x > 0) ? 1 : 0.0;
}

/*
 * This method represent the hiperbolic tangent function.
 */
double math_tanh_fx(double x) {
	return tanh(x);
}

/*
 * This method represent the derivative of the hiperbolic tangent function.
 */
double math_tanh_dx(double x) {
	return 1 - pow(tanh(x), 2);
}

/*
 * This method represent the simplest linear function. The math resolves to:
 *
 * f(x) = x
 */
double math_linear_fx(double x) {
	return x;
}

/*
 * This method represent the derivative of the simplest math function, which
 * it is just a constant: 1.
 */
double math_linear_dx(double x) {
	return 1;
}

/*
 * Nicely prints an array pointer as a matrix of size mxn.
 */
void math_printm(float* A, size_t m, size_t n) {
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			printf("%10.4f", A[i * n + j]);
			if (j + 1 < n) {
				printf("  ");
			}
		}
		printf("\n");
	}
	printf("\n");
	fflush (stdout);
}

/*
 * Nicely prints an array pointer as a transpose matrix.
 */
void math_printmt(float* A, size_t m, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < m; j++) {
			printf("%7.4f", A[i * m + j]);
			if (j + 1 < m) {
				printf(",  ");
			}
		}
		printf("\n");
	}
	printf("\n");
	fflush (stdout);
}

/*
 * Sets the value of an array pointer (which represents a matrix mxn) from
 * [-1, 1] using the uniform distribution.
 */
void math_fill(float* A, size_t m, size_t n) {
	math_fillr(A, m, n, 1);
}

/*
 * Sets the value of an array pointer (which represents a matrix mxn) from
 * [-base, base] using the uniform distribution.
 */
void math_fillr(float* A, size_t m, size_t n, const double base) {
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			//generate a random from [-base, 0, base]
			A[i * n + j] = (float) (-base + (2 * base) * (rand() / (double) RAND_MAX));
		}
	}
}

/*
 * Applies the given function f, adding the bias parameter, to the values of
 * v and stores the result in the same v vector. The math resolves to:
 *
 * v = f(v + bias)
 */
void math_apply_vs(float* v, size_t d, double (*f)(double x), float bias) {
	
	if (bias == 0.0) {
		for (size_t i = 0; i < d; i++) {
			v[i] = (float) f(v[i]);
		}
	} else {
		for (size_t i = 0; i < d; i++) {
			v[i] = (float) f(v[i] + bias);
		}
	}
}

/*
 * Applies the given function f, adding the bias parameter at the ith position
 * of v, to the values of v and stores the result in the same v vector.
 * The math resolves to:
 *
 * V(i) = f(V(i) + bias(i))
 */
void math_apply_vv(float* v, size_t d, double (*f)(double x), float* bias) {
	
	for (size_t i = 0; i < d; i++) {
		v[i] = (float) f(v[i] + bias[i]);
	}
}

/*
 * Applies the given function f to the matrix A, adding the bias parameter at
 * the jth column of A, and stores the result in the same A matrix. The math
 * resolves to:
 *
 * A(i,j) = f(A(i,j) + bias(j))
 *
 */
void math_apply_mv(float* A, size_t m, size_t n, double (*f)(double x), float* bias) {
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			A[i * n + j] = (float) f(A[i * n + j] + bias[j]);
		}
	}
}

/*
 * Subtracts the b vector to the a vector and stores the result in the
 * c vector. The math resolves to:
 *
 * c = a - b
 */
void math_vector_subtraction(float* a, float* b, float* c, size_t d) {
	for (size_t i = 0; i < d; i++) {
		c[i] = a[i] - b[i];
	}
}

/*
 * Perform element wise multiplication between a and b, and stores the result
 * in the c vector.
 */
void math_vector_elewise_mult(float* a, float* b, float* c, size_t d) {
	for (size_t i = 0; i < d; i++) {
		c[i] = a[i] * b[i];
	}
}

/*
 * Scales the a vector using the given scalar value.
 */
void math_vector_scale(float* a, float scalar, size_t d) {
	for (size_t i = 0; i < d; i++) {
		a[i] = a[i] * scalar;
	}
}

/*
 * Assigns the value of val to all the elements of the a vector.
 */
void math_vector_values(float* a, float val, size_t d) {
	for (size_t i = 0; i < d; i++) {
		a[i] = val;
	}
}

/*
 * Copies the element of the src array into the dest array.
 */
void math_vector_copy(float* dest, float* src, size_t length) {
	for (size_t i = 0; i < length; i++) {
		dest[i] = src[i];
	}
}

/*
 * Returns 1 if the number of operations exceeds the GPU computing threshold,
 * 0 otherwise.
 */
int math_gpu_threshold_reached(size_t operations) {
	
	return operations >= (size_t) GPU_THRESHOLD;
}

/*
 * Perform the dot product on a and b.
 */
double blas_dot(float* a, float* b, size_t d) {
	
	double sum = 0.0;
	
	for (size_t i = 0; i < d; i++) {
		sum += a[i] * b[i];
	}
	
	return sum;
}

//BLAS Level 3
/*
 * This method was inspired by the popular BLAS gemm function. It perform a
 * general matrix by matrix multiplication in which A and B can have different
 * sizes, and stores the result on the C matrix. If A and B are big enough that
 * the total operations exceeds the GPU threshold, then the computation will be
 * performed in the GPU. Otherwise the CPU is used instead by using a modified
 * version of the algorithm implemented in the GNU GSL library: gsl_blas_dgmm.
 * The math resolves to:
 *
 * C[mxp] = A[mxn] * B[nxp]
 */
void blas_gemm(float* A, float* B, float* C, size_t m, size_t n, size_t p) {
	
	if (math_gpu_threshold_reached(m * n * p)) {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		
		cublasStatus_t status = cublasXtSgemm(cublasXth, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha,
				B, p, A, n, &beta, C, p);
		
		if (status != CUBLAS_STATUS_SUCCESS) {
			
			fprintf(stderr, "Error: %d\n", status);
			fflush (stderr);
			exit (EXIT_FAILURE);
		}
	} else {
		math_vector_values(C, 0.0, m * p);
		
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < n; j++) {
				const float pivot = A[i * n + j];
				for (size_t k = 0; k < p; k++) {
					C[i * p + k] += pivot * B[j * p + k];
				}
			}
		}
	}
}

//Custom BLAS function
/*
 * This method is similar to blas_gemm, but it operates on the transpose
 * of the B matrix. The strategy for determining if GPU of CPU is the same
 * as blas_gemm. The math resolves to:
 *
 * C[mxp] = A[mxn] * T(B[pxn])
 */
void blas_gemmt(float* A, float* B, float* C, size_t m, size_t n, size_t p) {
	
	if (math_gpu_threshold_reached(m * n * p)) {
		const float alpha = 1.0f;
		const float beta = 0.0f;
		
		cublasStatus_t status = cublasXtSgemm(cublasXth, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha,
				B, n, A, n, &beta, C, p);
		
		if (status != CUBLAS_STATUS_SUCCESS) {
			
			fprintf(stderr, "Error: %d\n", status);
			fflush (stderr);
			exit (EXIT_FAILURE);
		}
	} else {
		for (size_t i = 0; i < m; i++) {
			for (size_t k = 0; k < p; k++) {
				double sum = 0.0;
				for (size_t j = 0; j < n; j++) {
					sum += A[i * n + j] * B[k * n + j];
				}
				C[i * p + k] = sum;
			}
		}
	}
}

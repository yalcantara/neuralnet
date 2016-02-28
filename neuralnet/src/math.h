/*
 * math.h
 *
 *  Created on: Jan 23, 2016
 *      Author: yaison
 */

#ifndef MATH_H_
#define MATH_H_

#include <cublasXt.h>

cublasXtHandle_t cublasXth;

typedef struct Func {
		double (*fx)(double x);
		double (*dx)(double x);
} Func;

double math_sigmoid_fx(double x);
double math_sigmoid_dx(double x);

double math_rectifier_fx(double x);
double math_rectifier_dx(double x);

double math_tanh_fx(double x);
double math_tanh_dx(double x);

double math_linear_fx(double x);
double math_linear_dx(double x);

void math_printm(float* A, size_t m, size_t n);
void math_printm(float* A, size_t m, size_t n);
void math_fill(float* A, size_t m, size_t n);
void math_fillr(float* A, size_t m, size_t n, const double base);

void math_apply_vs(float* v, size_t d, double (*fx)(double x), float bias);
void math_apply_vv(float* v, size_t d, double (*fx)(double x), float* bias);
void math_apply_mv(float* A, size_t m, size_t n, double (*fx)(double x), float* bias);

void math_vector_subtraction(float* a, float* b, float* c, size_t d);
void math_vector_elewise_mult(float* a, float* b, float* c, size_t d);
void math_vector_scale(float* a, float scalar, size_t d);
void math_vector_values(float* a, float val, size_t d);
void math_vector_copy(float* dest, float* src, size_t length);

int math_gpu_threshold_reached(size_t operations);
//BLAS Level 1
double blas_dot(float* a, float* b, size_t d);

//BLAS Level 3
void blas_gemm(float* A, float* B, float* C, size_t m, size_t n, size_t p);
void blas_gemmt(float* A, float* B, float* C, size_t m, size_t n, size_t p);

#endif /* MATH_H_ */

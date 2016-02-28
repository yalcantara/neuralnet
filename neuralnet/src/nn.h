/*
 * nn.h
 *
 *  Created on: Jan 23, 2016
 *      Author: yaison
 */

#ifndef NN_H_
#define NN_H_

#include "math.h"

typedef enum Regularization {
	L1, L2, NONE
} Regularization;

typedef struct Layer {
		
		size_t in;
		size_t out;
		float* bias;
		float* W;
		Func* func;
} Layer;

typedef struct Network {
		size_t in;
		size_t out;
		Layer** layers;
		int deepness;
		
} Network;

void error_gradient(float* A, float* E, float* grad, size_t m, size_t n, size_t p);
float nn_square_cost(Network* net, float* X, float* Y, size_t m, float* work);

Layer* nn_layer_create(size_t in, size_t out, Func* func);
size_t nn_layer_get_params(Layer* layer, float* dest);
size_t nn_layer_set_params(Layer* layer, float* params);
size_t nn_layer_params_size(Layer* layer);
size_t nn_layer_subtract_params(Layer* layer, float lambda, float* params);
size_t nn_layer_work_size(Layer* layer, size_t m);
void nn_layer_free(Layer* layer);
void nn_layer_print(Layer* layer);
void nn_layer_activate(Layer* layer, float* X, size_t m, float* G);
void nn_layer_gradient(Layer* layer, float* X, size_t m, float* G);
void nn_layer_gadient_descent(Layer* layer, float* X, float* Y, size_t m, float alpha, float* grad,
		float* work);

Network* nn_network_create(size_t in, int deepness, size_t out, Func** func);
size_t nn_network_get_params(Network* net, float* dest);
size_t nn_network_set_params(Network* net, float* params);
size_t nn_network_params_size(Network* net);
size_t nn_network_work_size(Network* net, size_t m);
void nn_network_gdescent(Network* net, float* X, float* Y, size_t m, float alpha, float lambda,
		Regularization reg, int iter);
float* nn_network_activate(Network* net, float* X, float* G, size_t m, float* work);
void nn_network_backprop(Network* net, float* X, float* Y, size_t m, float* grad, float lambda,
		Regularization reg, float** A, float** adx, float** error);
void nn_network_free(Network* net);
void nn_network_rand(Network* net);
void nn_network_print(Network* net);

#endif /* NN_H_ */

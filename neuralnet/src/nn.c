/*
 * nn.c
 *
 *  Created on: Jan 23, 2016
 *      Author: yaison
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "data.h"
#include "nn.h"

//Layers-----------------------------------------------------------------------

/*
 * Creates a Layer struct with the # of input, # of outputs and activation
 * function given by the in, out, and func parameters respectively.
 */
Layer* nn_layer_create(size_t in, size_t out, Func* func) {
	
	Layer* l = malloc(sizeof(Layer));
	//A row-mayor matrix that stores the weight for each neuron.
	//Each row at index i is the vector w (weight) for the i neuron.
	
	l->bias = calloc(out, sizeof(float));
	l->W = calloc(out * in, sizeof(float));
	l->in = in; //columns
	l->out = out; //rows
	l->func = func;
	
	//math_fill(layer->bias, out, 1);
	//math_fill(layer->W, out, in);
	
	return l;
}

/*
 * Deallocate the given Layer struct.
 */
void nn_layer_free(Layer* layer) {
	
	free(layer->bias);
	free(layer->W);
	free(layer);
}

/*
 * Sets a uniform random values from -1 to 1 to the given Layer parameters.
 */
void nn_layer_rand(Layer* layer) {
	math_fillr(layer->bias, layer->out, 1, 1);
	math_fillr(layer->W, layer->out, layer->in, 1);
}

/*
 * Copies the values of the given Layer's parameters into the dest pointer.
 * It goes neuron by neuron, and by writing first the bias unit and then the neuron's parameters.
 *
 * e.g,
 * bias1 | na_p1 | na_p2 | na_p3
 * bias2 | nb_p1 | nb_p2 | nb_p3
 *
 * Where bias[i] is the bias unit for the ith neuron, and n[i]_p[j] is the
 * value for the ith neuron and parameter j.
 */
size_t nn_layer_get_params(Layer* layer, float* dest) {
	size_t m = layer->out;
	size_t n = layer->in;
	
	float* W = layer->W;
	float* bias = layer->bias;
	
	size_t idx = 0;
	for (size_t i = 0; i < m; i++) {
		dest[idx] = bias[i];
		idx++;
		//bias done, now the parameters for the ith neuron
		for (size_t j = 0; j < n; j++) {
			dest[idx] = W[i * n + j];
			idx++;
		}
	}
	return idx;
}

/*
 * Copies the values form the given params pointer to the layer internal
 * parameters (including bias). This method is analogous to the
 * nn_layer_get_params function and assumes that the parameters are layout as
 * the nn_layer_get_params function stored it.
 *
 */
size_t nn_layer_set_params(Layer* layer, float* params) {
	size_t m = layer->out;
	size_t n = layer->in;
	
	float* W = layer->W;
	float* bias = layer->bias;
	
	size_t idx = 0;
	for (size_t i = 0; i < m; i++) {
		bias[i] = params[idx];
		idx++;
		for (size_t j = 0; j < n; j++) {
			W[i * n + j] = params[idx];
			idx++;
		}
	}
	
	return idx;
}

/*
 * This is a convenient method for updating the parameters of a given layer. It
 * subtract the values in the params pointer, scaled by alpha, to the given
 * current layer parameters.
 *
 * The math resolves to: θ = θ - α*Θ.
 *
 * Where:
 *  θ is the given layer parameters.
 * 	α is the given alpha value.
 * 	Θ the given params values.
 */
size_t nn_layer_subtract_params(Layer* layer, float alpha, float* params) {
	size_t m = layer->out;
	size_t n = layer->in;
	
	float* W = layer->W;
	float* bias = layer->bias;
	
	size_t idx = 0;
	if (alpha == 1.0f) {
		//we are using this algorithm for performance reasons, the idea is:
		//if alpha is 1, there is no need to multiply the parameters with it.
		for (size_t i = 0; i < m; i++) {
			bias[i] -= params[idx];
			idx++;
			for (size_t j = 0; j < n; j++) {
				W[i * n + j] -= params[idx];
				idx++;
			}
		}
	} else {
		size_t idx = 0;
		for (size_t i = 0; i < m; i++) {
			bias[i] -= alpha * params[idx];
			idx++;
			for (size_t j = 0; j < n; j++) {
				W[i * n + j] -= alpha * params[idx];
				idx++;
			}
		}
	}
	
	return idx;
}

/*
 * Computes the total size of the parameters (including the bias unit)
 * of a given layer.
 */
size_t nn_layer_params_size(Layer* layer) {
	size_t weigths = layer->in * layer->out;
	size_t biases = layer->out;
	
	return weigths + biases;
}

/*
 * This method computes the total memory size (x float) required to
 * activate the given layer.
 */
size_t nn_layer_work_size(Layer* layer, size_t m) {
	return m * layer->out;
}

/*
 * Activates the given layer and stores the result in the A vector.
 * The minimum required length of A can be computed by using the
 * function nn_layer_work_size.
 */
void nn_layer_activate(Layer* layer, float* X, size_t m, float* A) {
	blas_gemmt(X, layer->W, A, m, layer->in, layer->out);
	math_apply_mv(A, m, layer->out, layer->func->fx, layer->bias);
}

/*
 * Computes the gradient of the layer by applying the derivative of the
 * activation function.
 */
void nn_layer_gradient(Layer* layer, float* X, size_t m, float* G) {
	blas_gemmt(X, layer->W, G, m, layer->in, layer->out);
	math_apply_mv(G, m, layer->out, layer->func->dx, layer->bias);
}

/*
 * Computes the "delta rule" for the given A (activation) pointer and
 * E (error) pointer.
 *
 * The math resolves to: Δ = (1 / m) * δ(trans) * [1 a]
 *
 * where:
 * 	Δ are the new update values.
 * 	m the number of instances (training cases).
 * 	δ(trans) the transpose of the error.
 * 	[1 a] is the activation with a column full of 1 inserted at at index 0.
 *
 * note: EG for error gradient, and output pointer.
 */
void error_gradient(float* A, float* E, float* EG, size_t m, size_t n, size_t p) {
	
	//since the activation is A* T(W), the resulting matrix is pxn and not mxp.
	//we add the (p+1) because of the bias term.
	const size_t size = (p + 1) * n;
	
	math_vector_values(EG, 0.0f, size);
	
	//for efficiency, we are going to accumulate the gradient by going row->column
	//(the same as row-major), which is a sequential read/write to memory.
	for (size_t k = 0; k < p; k++) {
		for (size_t i = 0; i < m; i++) {
			double error = E[i * p + k];
			//the bias is the first element in each row.
			//+1 because the bias is not included in n.
			EG[k * (n + 1)] += error;
			
			for (size_t j = 0; j < n; j++) {
				//+1 because the bias is the first element in each row.
				EG[k * (n + 1) + j + 1] += error * A[i * n + j];
			}
		}
	}
	
	math_vector_scale(EG, 1.0 / m, size);
}

/*
 * Applies the Gradient Descent algorithm to a given layer. The size of the
 * work array must be at least nn_layer_work_size(layer, m).
 */
void nn_layer_gadient_descent(Layer* layer, float* A, float* Y, size_t m, float alpha, float* grad,
		float* work) {
	
	nn_layer_activate(layer, A, m, work);
	
	size_t n = layer->in;
	size_t p = layer->out;
	
	//In the same work matrix let us store the difference.
	//We can treat the matrixes as a single vector (row-major layout).
	math_vector_subtraction(work, Y, work, m * p);
	
	//work is our error matrix: A - Y
	error_gradient(A, work, grad, m, n, p);
	
	nn_layer_subtract_params(layer, alpha, grad);
	
}

/*
 * Nicely prints the parameters of a given layer.
 */
void nn_layer_print(Layer* la) {
	int m = la->out;
	int n = la->in;
	for (int i = 0; i < m; i++) {
		
		printf("%8.4f  |  ", la->bias[i]);
		for (int j = 0; j < n; j++) {
			printf("%7.4f", la->W[i * n + j]);
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
 * This method creates a Network struct with input, layers, output units and
 * activation functions given by the in, deepness, out and func parameters.
 * For simplicity, each layer (except the output layer) will have the same
 * number of input and output units. If different in-output is required between
 * the hidden layers, then it have to be created manually. The activation
 * function array must be equal to the number of layers (deepness). Note that
 * each layer may have different activation function.
 */
Network* nn_network_create(size_t in, int deepness, size_t out, Func** func) {
	Network* net = malloc(sizeof(Network));
	net->in = in;
	net->deepness = deepness;
	net->out = out;
	net->layers = malloc(sizeof(Layer*) * deepness);
	
	if (deepness == 1) {
		net->layers[0] = nn_layer_create(in, out, func[0]);
	} else {
		const int lastIdx = deepness - 1;
		
		//all hidden layers will have the same number of input and
		//output units.
		for (int i = 0; i < lastIdx; i++) {
			net->layers[i] = nn_layer_create(in, in, func[i]);
		}
		//the last layer (output layer) is the only one that has
		//a different number of output units.
		net->layers[lastIdx] = nn_layer_create(in, out, func[lastIdx]);
	}
	
	return net;
}

/*
 * Deallocate the Network struct.
 */
void nn_network_free(Network* net) {
	int L = net->deepness;
	for (int l = 0; l < L; l++) {
		nn_layer_free(net->layers[l]);
	}
	free(net->layers);
	free(net);
}

/*
 * Sets the parameters of the given network random values using the uniform
 * distribution from -1 to 1.
 */
void nn_network_rand(Network* net) {
	for (int l = 0; l < net->deepness; l++) {
		nn_layer_rand(net->layers[l]);
	}
}

/*
 * Copies the given netowork's parameters into a array. This method  works
 * by calling nn_layer_get_params for each layer and sending the dest pointer where
 * it left off +1.
 */
size_t nn_network_get_params(Network* net, float* dest) {
	
	int L = net->deepness;
	size_t size = 0;
	for (int l = 0; l < L; l++) {
		Layer* layer = net->layers[l];
		size += nn_layer_get_params(layer, dest + size);
	}
	
	return size;
}

/*
 * This is analogous version of nn_network_get_params but copies from the given
 * params pointer to the Network internal parameters. It does so by calling
 * nn_layer_set_params for each layer and sending the params pointer where
 * it left off +1.
 */
size_t nn_network_set_params(Network* net, float* params) {
	
	int L = net->deepness;
	size_t size = 0;
	for (int l = 0; l < L; l++) {
		Layer* layer = net->layers[l];
		size += nn_layer_set_params(layer, params + size);
	}
	
	return size;
}

/*
 * Subtracts the values in the given params array to the Networks internal parameters.
 * This method works by calling nn_layer_subtract_params for each layer.
 *
 * The math resolves to: θ = θ - α*Θ.
 *
 * Where:
 *  θ is the given network parameters.
 * 	α is the given alpha value.
 * 	Θ the given params values.
 */
void nn_network_subtract_params(Network* net, float alpha, float* params) {
	
	const int L = net->deepness;
	Layer** layers = net->layers;
	
	Layer* crt;
	size_t idx = 0;
	for (int i = 0; i < L; i++) {
		crt = layers[i];
		
		float* layerParams = params + idx;
		nn_layer_subtract_params(crt, alpha, layerParams);
		idx += nn_layer_params_size(crt);
	}
}

/*
 * Computes the total parameters in the given network. It does so by calling
 * nn_layer_params_size for each layer and accumulating the result.
 */
size_t nn_network_params_size(Network* net) {
	size_t size = 0;
	
	int L = net->deepness;
	for (int l = 0; l < L; l++) {
		size += nn_layer_params_size(net->layers[l]);
	}
	
	return size;
}

/*
 * Computes the total parameters needed to fully activate a given network.
 * It does so by computing the work size of the largest layer and multiplying
 * it by 2.
 */
size_t nn_network_work_size(Network* net, size_t m) {
	int L = net->deepness;
	Layer** layers = net->layers;
	size_t highest = nn_layer_work_size(layers[0], m);
	
	for (int l = 1; l < L; l++) {
		size_t size = nn_layer_work_size(layers[l], m);
		highest = (size > highest) ? size : highest;
	}
	
//we need x2 the length of the data by the largest layer output.
	return 2 * m * highest;
}

/*
 * Nicely prints the network parameters.
 */
void nn_network_print(Network* net) {
	int L = net->deepness;
	for (int l = 0; l < L; l++) {
		printf("Layer %2d\n", (l + 1));
		nn_layer_print(net->layers[l]);
	}
}

/*
 * Fully activates the given network by starting at the first layer and then
 * propagating the activations to the next (deeper) layer up to the last one.
 *
 * Parameters:
 * net: the given Network structure to make the activation.
 * A: the activation matrix or input.
 * C: the destination matrix or output (where the result will be saved).
 * m: the number of rows of the A matrix.
 * work: a buffer to store the intermediate activations (between layers).
 *
 *
 * Note: The size of the work array must be at least the returned value of nn_network_work_size(net, m).
 */
float* nn_network_activate(Network* net, float* A, float* C, size_t m, float* work) {
	
	int L = net->deepness;
	
	if (L == 1) {
		if (C == NULL) {
			//If no output is given, let just leave it in the work buffer.
			nn_layer_activate(net->layers[0], A, m, work);
			return work;
		}
		nn_layer_activate(net->layers[0], A, m, C);
		return C;
	}
	Layer* layer;
	
	//work1 contains the first half of the work buffer
	//and work2 the second half. We need 2 halves: 1 for the
	//intermediate activation and the 2nd for storing the result
	//and as input onto the next layer.
	float* work1 = work;
	float* work2 = work + (nn_network_work_size(net, m) / 2);
	float* temp;
	
	layer = net->layers[0];
	nn_layer_activate(layer, A, m, work1);
	
	int lastIdx = L - 1;
	for (int l = 1; l < lastIdx; l++) {
		layer = net->layers[l];
		nn_layer_activate(layer, work1, m, work2);
		
		//here is where the shift occurs: since we did an activation
		// from layer L(l) to layer L(l+1), the output of a layer L(l) becomes
		//the input of L(l+1), and the input of L(l) will be the output of L(l+1).
		temp = work1;
		work1 = work2;
		work2 = temp;
	}
	
	layer = net->layers[lastIdx];
	if (C == NULL) {
		//again, if C wasn't given let us just leave the result at the
		//second half of work (work2).
		nn_layer_activate(layer, work1, m, work2);
		return work2;
	}
	nn_layer_activate(layer, work1, m, C);
	return C;
	
}

/*
 * Computes the squared error for the activation of a given network with
 * respect some expected output Y.
 *
 * the math resolves to:
 * J = (1 / (2*m)) * Σ(Y - h(X))^2
 *
 * where:
 * X: the input matrix.
 * Y: the desired (or target) output matrix.
 * m: is the number of rows of X and Y (must be the same for both of them).
 * h(X): is the hypothesis output given by the activation of the network.
 *
 *
 * note: The size of the work array must be at least nn_network_work_size(net, m).
 */
float nn_square_cost(Network* net, float* X, float* Y, size_t m, float* work) {
	
	//A for Activation
	//A isn't allocated by the method. It's just a pointer to work, where the activation was left of.
	float* A = nn_network_activate(net, X, NULL, m, work);
	
	float sum = 0.0;
	
	size_t total = m * net->out;
	for (size_t i = 0; i < total; i++) {
		sum += pow(A[i] - Y[i], 2);
	}
	
	float j = 1.0 / (2.0 * m) * sum;
	
	return j;
}

/*
 * Applies the Gradient Descent algorithm to the given network and perform the
 * weight updates. For efficiency, the parameter iter can be passed, to reduce
 * subsequent calls to this method. Note that after calling this method, it is
 * perfectly fine to re-called again and again (but it is advised to use the
 * iter parameter instead).
 *
 * The gradients are computed using the back-propagation algorithm which computes
 * the gradient for each parameter in each layer. For more information on how the
 * back-propagation algorithm works, please check out:
 *
 * https://en.wikipedia.org/wiki/Backpropagation
 *
 * Parameters
 * net: the Network struct to which compute the gradient on.
 * X: the input matrix.
 * Y: the desired output or labeled training matrix.
 * m: the number of rows for X, which has to be the same for Y.
 * alpha: the learning rate (for most purpose 1 should be ok).
 * lambda: the regularization multiplier if it should be used (or 0 otherwise).
 * reg: the type of regularization to be used (L1 or L2 are supported, or NULL
 * 		if no regularization should be be applied).
 *
 *
 */
void nn_network_gdescent(Network* net, float* X, float* Y, size_t m, float alpha, float lambda,
		Regularization reg, int iter) {
	
	const int L = net->deepness;
	const size_t out = net->out;
	Layer** layers = net->layers;
	
	Layer* crt = layers[0];
	
	//The grad (network update for parameters).
	size_t params_size = nn_network_params_size(net);
	float* grad = malloc(sizeof(float) * params_size);
	
	//Activation buffer
	float** A = malloc(sizeof(float*) * (L + 1));
	A[0] = X;
	for (int l = 0; l < L; l++) {
		crt = layers[l];
		A[l + 1] = malloc(sizeof(float) * m * crt->out);
		
	}
	
	//=====================================================================
	
	//A' buffer
	float** adx = malloc(sizeof(float*) * (L - 1));
	for (int l = 0; l < L - 1; l++) {
		crt = layers[l];
		adx[l] = malloc(sizeof(float) * m * crt->out);
	}
	
	//=====================================================================
	// Error buffer
	float** error = malloc(sizeof(float*) * L);
	error[L - 1] = malloc(sizeof(float) * m * out);
	for (int l = L - 2; l >= 0; l--) {
		crt = layers[l];
		const size_t p = crt->out;
		error[l] = malloc(sizeof(float) * m * p);
	}
	
	//Now let's back-propagate.
	for (int i = 0; i < iter; i++) {
		nn_network_backprop(net, X, Y, m, grad, lambda, reg, A, adx, error);
		nn_network_subtract_params(net, alpha, grad);
	}
	
	free(grad);
	for (int l = 0; l < L; l++) {
		free(A[l + 1]);
	}
	free(A);
	for (int l = 0; l < L - 1; l++) {
		free(adx[l]);
	}
	free(adx);
	for (int l = 0; l < L; l++) {
		free(error[l]);
	}
	free(error);
	//=========================================================================
}

/*
 * Computes the gradient of the parameters of a given network. This method uses
 * the back-propagation algorithm, but do not perform the weight updates on to
 * the network, it instead saves the gradient in the grad parameter. For more
 * information on how the back-propagation algorithm works please check out:
 *
 * https://en.wikipedia.org/wiki/Backpropagation.
 *
 * Parameters
 * net: the Network struct to which compute the gradient on.
 * X: the input matrix.
 * Y: the desired output or labeled training matrix.
 * m: the number of rows for X, which has to be the same for Y.
 * grad: the array on where the gradients will be stored.
 * lambda: the regularization multiplier if it should be used (or 0 otherwise).
 * reg: the type of regularization to be used (L1 or L2 are supported, or NULL
 * 		if no regularization should be be applied).
 * A: an activation for each layer (computed by this method).
 * adx: the gradient for each layer activation (computed by this method).
 * error: the error for each layer (computed by this method);
 */
void nn_network_backprop(Network* net, float* X, float* Y, size_t m, float* grad, float lambda,
		Regularization reg, float** A, float** adx, float** error) {
	
	// let's assume only the hidden layers and the output layer.
	const int L = net->deepness;
	const size_t out = net->out;
	Layer** layers = net->layers;
	
	Layer* crt = layers[0];
	
	//=====================================================================
	//A[0] must be equal to X
	// our layers starts with the hidden layer having 1 less index than the
	// A (a) array.
	for (int l = 0; l < L; l++) {
		crt = layers[l];
		float* x = A[l];
		
		float* C = A[l + 1];
		nn_layer_activate(crt, x, m, C);
	}
	
	//=====================================================================
	// for a' (adx) we don't need the last layer, making indexing a bit awkward.
	for (int l = 0; l < L - 1; l++) {
		crt = layers[l];
		float* X_l = A[l];
		//G for gradient ^_^
		float* G = adx[l];
		nn_layer_gradient(crt, X_l, m, G);
		
	}
	
	//=====================================================================
	// The total elements for the error (δ), is the same as the hidden +
	// output layers
	const size_t Alength = L + 1;
	
	math_vector_subtraction(A[Alength - 1], Y, error[L - 1], m * out);
	//let's go backwards, starting at L-2 because we already
	//set L-1 to the error array.
	for (int l = L - 2; l >= 0; l--) {
		crt = layers[l];
		const size_t n = layers[l + 1]->out;
		const size_t p = crt->out;
		
		blas_gemm(error[l + 1], layers[l + 1]->W, error[l], m, n, p);
		math_vector_elewise_mult(error[l], adx[l], error[l], m * p);
	}
	
	//=====================================================================
	
	size_t gradIdx = 0;
	for (int l = 0; l < L; l++) {
		crt = layers[l];
		
		const size_t n = crt->in;
		const size_t p = crt->out;
		
		float* A_l = A[l];
		float* E = error[l];
		float* EG = grad + gradIdx;
		
		error_gradient(A_l, E, EG, m, n, p);
		
		if (lambda != 0.0) {
			//Let's apply regularization. Currently we only support L1 and L2 regularization.
			if (reg == L1) {
				for (size_t i = 0; i < p; i++) {
					for (size_t j = 1; j < n; j++) {
						EG[i * (n + 1) + j + 1] += lambda / 2.0 * m;
					}
				}
			} else if (reg == L2) {
				float* W = crt->W;
				for (size_t i = 0; i < p; i++) {
					for (size_t j = 0; j < n; j++) {
						EG[i * (n + 1) + j + 1] += lambda / m * W[i * n + j];
					}
				}
			}
		}
		
		gradIdx += nn_layer_params_size(crt);
	}
}

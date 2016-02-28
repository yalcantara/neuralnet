#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include "math.h"
#include "nn.h"
#include "data.h"

Matrix* X;
Matrix* Y;

Matrix* X_train;
Matrix* Y_train;

Matrix* X_cross;
Matrix* Y_cross;

Matrix* X_test;
Matrix* Y_test;

/*
 * This method loads the iris dataset and creates the appropriate matrices
 * for training, cross validation and testing.
 */
void iris() {
	String* raw = ffull("files/iris.data");
	
	Grid* g = grid_create(raw->value, ',');
	
	if (g) {
		str_free(raw);
		Mapper* map = map_create(g);
		if (map) {
			Matrix* mtr = mtr_create_grid(g, map, 1);
			mtr_shuffle(mtr);
			
			X = mtr_col_slct(mtr, 0, 4);
			Y = mtr_col_slct(mtr, 4, 7);
			
			//Let's split the data set into:
			//60% for training
			//20% for cross validation
			//20% for the final test
			size_t m = Y->m;
			size_t trainIdx = (size_t)(m * 0.6);
			size_t testStartIdx = (size_t)(m * 0.8);
			X_train = mtr_row_slct(X, 0, trainIdx);
			Y_train = mtr_row_slct(Y, 0, trainIdx);
			
			X_cross = mtr_row_slct(X, trainIdx, testStartIdx);
			Y_cross = mtr_row_slct(Y, trainIdx, testStartIdx);
			
			X_test = mtr_row_slct(X, testStartIdx, m);
			Y_test = mtr_row_slct(Y, testStartIdx, m);
			
			mtr_free(mtr);
		}
		map_free(map);
	}
	
	grid_free(g);
}

int main() {
	srand((unsigned) time(NULL));
	
	iris();
	
	size_t m = Y_train->m;
	size_t n = X_train->n;
	size_t p = Y_train->n;
	size_t m_cross = Y_cross->m;
	size_t m_test = Y_test->m;
	
	//this is a very simple network architecture with 1 hidden layer and 1 output layer.
	int deepness = 2;
	
	float* A_train = X_train->values;
	float* B_train = Y_train->values;
	
	float* A_cross = X_cross->values;
	float* B_cross = Y_cross->values;
	
	float* A_test = X_test->values;
	float* B_test = Y_test->values;
	
	//Since we are doing classification, it makes sense to use the logistic function
	Func** funcs = malloc(sizeof(Func*) * deepness);
	for (int i = 0; i < deepness; i++) {
		funcs[i] = malloc(sizeof(Func));
		funcs[i]->fx = math_sigmoid_fx;
		funcs[i]->dx = math_sigmoid_dx;
	}
	
	Network* net = nn_network_create(n, deepness, p, funcs);
	nn_network_rand(net);
	printf("Network init\n\n");
	fflush (stdout);
	
	size_t size = nn_network_work_size(net, Y->m);
	//work is large enough for cross the full batch, training, cross validation and testing.
	float* work = malloc(sizeof(float) * size);
	
	size_t params_size = nn_network_params_size(net);
	float* params = malloc(sizeof(float) * params_size);
	const int iter = 1000;
	
	//We are going to train using severals lambda parameters. Then we are
	//going to choose the lambda that produced the min squared error for testing.
	//Note that we are going to train using the training dataset and test the lambda
	//parameter using the cross validation set.
	float reg[] = { 0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0 };
	
	float minJ = nn_square_cost(net, A_cross, B_cross, m_cross, work);
	float minLambda = 0.0;
	nn_network_get_params(net, params);
	for (int i = 0; i < 10; i++) {
		float lambda = reg[i];
		
		nn_network_gdescent(net, A_train, B_train, m, 1.0, lambda, L2, iter);
		
		float J = nn_square_cost(net, A_cross, B_cross, m_cross, work);
		if (J < minJ) {
			//let's save the min squared error and the network parameters
			minJ = J;
			minLambda = lambda;
			nn_network_get_params(net, params);
		}
		
		printf("Lambda: %8.4f  ->  %12.8f\n", lambda, J);
		fflush(stdout);
		nn_network_rand(net);
	}
	
	//let's set the parameters that produced the min error.
	nn_network_set_params(net, params);
	printf("\n\n");
	printf("Using lambda: %8.4f  for  J: %12.8f", minLambda, minJ);
	
	float* C = malloc(sizeof(float) * (m_test * p));
	nn_network_activate(net, A_test, C, m_test, work);
	
	//Let's test how our model can perform with unseen data (test dataset).
	//We are going to print 'ok' if the output is >= 0.5 when is the correct class (1) or
	//if it isn't (0) and our model output is < 0.5. Otherwise we are going to print 'wrong'.
	printf("\n\n");
	printf("Iris Setosa  |  Iris Versicolour  |  Iris Virginica\n");
	printf("---------------------------------------------------\n");
	int seok = 0;
	int veok = 0;
	int viok = 0;
	int count = m_test;
	for (size_t i = 0; i < count; i++) {
		float yse = B_test[i * 3];
		float yve = B_test[i * 3 + 1];
		float yvi = B_test[i * 3 + 2];
		
		float hse = C[i * 3];
		float hve = C[i * 3 + 1];
		float hvi = C[i * 3 + 2];
		
		if ((yse == 1.0 && hse >= 0.5) || (yse == 0.0 && hse < 0.5)) {
			printf("    ok    ");
			seok++;
		} else {
			printf("   wrong  ");
		}
		printf("   |     ");
		if ((yve == 1.0 && hve >= 0.5) || (yve == 0.0 && hve < 0.5)) {
			printf("    ok    ");
			veok++;
		} else {
			printf("   wrong  ");
		}
		printf("     |   ");
		if ((yvi == 1.0 && hvi >= 0.5) || (yvi == 0.0 && hvi < 0.5)) {
			printf("    ok    ");
			viok++;
		} else {
			printf("   wrong  ");
		}
		
		printf("\n");
	}
	
	printf("\n\n");
	
	printf("   Class         |   ok   |  Wrong  |   %% ok   |  %% Wrong                        \n");
	printf("------------------------------------------------------------\n");
	printf("Iris Setosa      |   %3d  |   %3d   |  %6.2f  | %6.2f\n", seok, (count - seok),
			seok / (float) count * 100, (count - seok) / (float) count * 100);
	printf("Iris Versicolour |   %3d  |   %3d   |  %6.2f  | %6.2f\n", veok, (count - veok),
			veok / (float) count * 100, (count - veok) / (float) count * 100);
	printf("Iris Virginica   |   %3d  |   %3d   |  %6.2f  | %6.2f\n", viok, (count - viok),
			viok / (float) count * 100, (count - viok) / (float) count * 100);
	fflush(stdout);
	
	return EXIT_SUCCESS;
}


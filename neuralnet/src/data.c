/*
 * data.c
 *
 *  Created on: Jan 25, 2016
 *      Author: yaison
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/types.h>
#include "data.h"

#ifndef NAN
#define NAN nan("")
#endif

/*
 * Creates a Matrix struct based on the Grid object provided. If the Grid
 * contains words, the resulting matrix will have a column for each of
 * them with values 0 or 1 in it's row if the grid contains the word
 * at the row index.
 */
Matrix* mtr_create_grid(Grid* g, Mapper* mapper, int scale) {
	
	size_t m = g->info->rows;
	
	size_t* missing = g->info->missing;
	
	size_t* sizes = mapper->sizes;
	char*** map = mapper->map;
	
	size_t cols = mapper->cols;
	
	size_t n = mtr_cols(map, sizes, missing, cols);
	
	//Let's construct the Matrix struct, now that we have the total number of columns
	//required to represent the given Grid into a Matrix.
	Matrix* matrix = mtr_create(m, n);
	float* values = matrix->values;
	
	size_t rows = g->info->rows;
	char*** body = g->body;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			char* val = body[i][j];
			//mtrcol holds the column index where the value should be stored.
			size_t mtrcol = to_mtr_col(map, sizes, missing, j, val);
			if (mtrcol == -1) {
				fprintf(stderr, "Could not create Matrix");
				fflush (stderr);
				return NULL;
			}
			
			size_t words = g->info->words[j];
			short discrete = g->info->discrete[j];
			if (val == NULL || words > 0) {
				//if it's a word, let just put 1.
				//if the rows doesn't have that particular word, it is defaulted to 0.
				values[i * n + mtrcol] = 1;
			} else {
				
				float v = atof(val);
				
				//We are currently not scaling discrete values because we want
				//to keep Grid columns that have only 0 and 1 (binary) intact.
				//TODO separate discrete and binary values and allow for scaling individually.
				if (discrete != 1 && scale) {
					float mean = g->info->mean[j];
					float stdev = g->info->stdev[j];
					v = (v - mean) / stdev;
				}
				values[i * n + mtrcol] = v;
			}
		}
	}
	
	return matrix;
}

/*
 * Computes the total number of rows a Matrix would have if constructed based
 * on a given grid. In this particular case, we are using char*** instead of
 * the Grid structure.
 */
size_t mtr_cols(char*** map, size_t* sizes, size_t* missing, size_t col) {
	size_t n = 0;
	for (size_t j = 0; j < col; j++) {
		if (map[j] == NULL) {
			//numeric type
			if (missing[j] == 0) {
				n++;
			} else {
				n += 2;
			}
		} else {
			n += sizes[j];
		}
	}
	return n;
}

/*
 * This method creates a Mapper struct, based on the given Grid struct.
 * A Mapper contains the required information for converting Grid columns to
 * the corresponding column of it's matrix.
 */
Mapper* map_create(Grid* g) {
	
	size_t n = g->info->columns;
	
	size_t* sizes = calloc(n, sizeof(size_t));
	char*** map = calloc(n, sizeof(char**));
	
	for (size_t j = 0; j < n; j++) {
		size_t numbers = g->info->numbers[j];
		size_t words = g->info->words[j];
		
		if (numbers == 0 && words == 0) {
			//empty column
			sizes[j] = 0;
			map[j] = NULL;
		} else if (numbers == 0 && words > 0) {
			//all words
			size_t l;
			char** arr = str_uniq(g, j, &l);
			sizes[j] = l;
			map[j] = arr;
			
		} else if (numbers > 0 && words == 0) {
			//all numbers
			sizes[j] = 0;
			map[j] = NULL;
		} else {
			//mixed, treated as words
			size_t l;
			char** arr = str_uniq(g, j, &l);
			sizes[j] = l;
			map[j] = arr;
		}
	}
	
	Mapper* mapper = malloc(sizeof(Mapper));
	mapper->cols = n;
	mapper->sizes = sizes;
	mapper->map = map;
	
	return mapper;
}

/*
 * This method deallocate the Mapper structure.
 */
void map_free(Mapper* mapper) {
	if (mapper) {
		char*** map = mapper->map;
		size_t cols = mapper->cols;
		size_t* sizes = mapper->sizes;
		
		if (map) {
			for (size_t i = 0; i < cols; i++) {
				size_t size = sizes[i];
				for (size_t j = 0; j < size; j++) {
					free(map[i][j]);
				}
				free(map[i]);
			}
		}
		
		if (sizes) {
			free(sizes);
		}
		
		free(mapper);
	}
}

/*
 * This method creates a GridInfo struct based on the given Grid. A GridInfo is
 * a struct which contains important information about the Grid for each
 * column, such as: maximum and minimum value (if numeric), number of missing
 * values, if it's numeric etc. Note that this method is called while the Grid
 * is being created, that is why it need the row and column parameters.
 */
GridInfo* grid_info(Grid* g, size_t rows, size_t cols) {
	
	char*** body = g->body;
	
	GridInfo* info = malloc(sizeof(GridInfo));
	info->max = malloc(sizeof(float) * cols);
	info->min = malloc(sizeof(float) * cols);
	info->mean = malloc(sizeof(float) * cols);
	info->stdev = malloc(sizeof(float) * cols);
	info->discrete = malloc(sizeof(short) * cols);
	info->missing = malloc(sizeof(size_t) * cols);
	info->numbers = malloc(sizeof(size_t) * cols);
	info->words = malloc(sizeof(size_t) * cols);
	
	for (int j = 0; j < cols; j++) {
		short discrete;
		float max = NAN;
		float min = NAN;
		size_t missing = 0;
		size_t numbers = 0;
		size_t words = 0;
		
		size_t discreteCount = 0;
		for (int i = 0; i < rows; i++) {
			char* val = body[i][j];
			if (val == NULL) {
				//easy case, val is NULL ^_^
				missing++;
			} else {
				char* end;
				float num = strtof(val, &end);
				if (val == end) {
					//not a number
					words++;
				} else {
					if (ceil(num) == num) {
						discreteCount++;
					}
					max = higher(max, num);
					min = lower(min, num);
					numbers++;
				}
			}
		}
		
		if (numbers > 0) {
			//let's see if all the numbers for the column j are all discrete
			discrete = discreteCount == numbers;
		} else {
			discrete = 0;
		}
		
		info->discrete[j] = discrete;
		info->max[j] = max;
		info->min[j] = min;
		info->missing[j] = missing;
		info->numbers[j] = numbers;
		info->words[j] = words;
		
	}
	info->rows = rows;
	info->columns = cols;
	
	printf("About to fill stdev\n");
	fflush (stdout);
	fill_stdev(info, g);
	
	return info;
}

/*
 * This method creates a Grid struct, based on the input dataset. It asumes
 * that each instance is separated by '\n' (lines) and delimited by the
 * given d parameter. A Grid is table like format for storing the tabular dataset.
 */
Grid* grid_create(char* raw, char d) {
	//Let's do a quick check to verify that the file has '\n'
	ssize_t lines = char_find(raw, '\n');
	if (lines == -1) {
		fflush (stdout);
		fprintf(stderr, "The input has no \n char.");
		fflush (stderr);
		return NULL;
	}
	
	size_t cols = char_count(raw, lines, d);
	size_t rows = char_count(raw, -1, '\n');
	cols++;
	rows++;
	
	if (cols < 2) {
		fflush (stdout);
		fprintf(stderr, "Error creating Grid. Less than 2 columns.");
		fflush (stderr);
		return NULL;
	}
	
	//TODO remove the hardcoded 10 value
	if (rows < 10) {
		fflush (stdout);
		fprintf(stderr, "Error creating Grid. Less than 10 rows.");
		fflush (stderr);
		return NULL;
	}
	
	Grid* g = malloc(sizeof(Grid));
	g->info = NULL;
	//We need to default it to NULL values.
	char*** body = calloc(rows, sizeof(char**));
	g->body = body;
	
	size_t start = 0;
	char* p = raw;
	for (int i = 0; i < rows; i++) {
		char** row = calloc(cols, sizeof(char*));
		
		for (int j = 0; j < cols; j++) {
			p = p + start;
			
			if (p[0] == '\n' && j == 0) {
				//at this point there is an empty line.
				//we are going to stop there
				
				fflush (stdout);
				fprintf(stderr, "\nBlank line detected at line %d.\n", (i + 1));
				fflush (stderr);
				grid_partial_free(g, i - 1, cols);
				free(row);
				return NULL;
			}
			
			//let's find the delimiter or line break '\n'
			ssize_t l;
			if (j + 1 < cols) {
				l = char_find(p, d);
			} else if (i + 1 < rows) {
				l = char_find(p, '\n');
			} else {
				l = strlen(p);
			}
			
			if (l == -1) {
				fflush (stdout);
				fprintf(stderr, "\nInvalid row. Row %d, detected at col %d.\n", (i + 1), (j + 1));
				fflush (stderr);
				grid_partial_free(g, i - 1, cols);
				free(row);
				return NULL;
			}
			
			//let's pack each value.
			size_t tl = trim_lenght(p, l);
			if (tl > 0) {
				char* col = char_create_arr(tl);
				trim(p, l, col, tl);
				
				row[j] = col;
			} else {
				row[j] = NULL;
			}
			start = l + 1;
		}
		
		//just for precaution, let's detect if there is an empty line, a line with no columns.
		size_t allNull = 1;
		for (int j = 0; j < cols; j++) {
			if (row[j] != NULL) {
				allNull = 0;
				break;
			}
		}
		
		if (allNull) {
			
			fflush (stdout);
			fprintf(stderr,
					"\nThere is a problem with the input, got all column values null for row %d.\n",
					(i + 1));
			fflush (stderr);
			grid_partial_free(g, i - 1, cols);
			free(row);
			return NULL;
		}
		
		body[i] = row;
	}
	
	printf("About to create grid info\n");
	fflush (stdout);
	g->info = grid_info(g, rows, cols);
	
	return g;
}

/*
 * This method finds a char in the given char array, stopping if found or if
 * it has reached the null character. Returns the index of the found char or
 * -1 if it couldn't found it.
 */
ssize_t char_find(char* str, char c) {
	
	char crt;
	for (size_t i = 0;; i++) {
		crt = str[i];
		if (crt == '\0') {
			return -1;
		}
		
		if (crt == c) {
			return i;
		}
	}
	
	return -1;
}

/*
 * This method searches the position where a given character is NOT found in
 * string. In other words, it returns the very first index (going sequentially)
 * where the char was NOT found.
 */
ssize_t char_not_find(char* str, char c) {
	
	char crt;
	for (size_t i = 0;; i++) {
		crt = str[i];
		if (crt == '\0') {
			return -1;
		}
		
		if (crt != c) {
			return i;
		}
	}
	
	return -1;
}

/*
 * This method does exactly what char_not_find do, but instead of going from
 * left to right, this method goes from right to left.
 */
ssize_t char_not_find_backwards(char* str, size_t length, char c) {
	char crt;
	for (size_t i = length - 1; i >= 0; i--) {
		crt = str[i];
		if (crt == 0) {
			return -1;
		}
		
		if (crt != c) {
			return i;
		}
	}
	
	return -1;
}

/*
 * Reads the file denoted by the given path entirely and stores it in memory.
 * Returns a String struct, with the all the characters within the file.
 */
String* ffull(char* path) {
	FILE* f = fopen(path, "r");
	
	if (f == NULL) {
		fprintf(stderr, "file not found at: %s\n", path);
		fflush (stderr);
		return NULL;
	}
	
	char c;
	
	//A trick to determine how many characters a file has
	size_t count = 0;
	while ((c = fgetc(f))) {
		if (c == EOF) {
			break;
		}
		count++;
	}
	
	char* content = char_create_arr(count);
	rewind(f);
	size_t r = fread(content, sizeof(char), count, f);
	
	fclose(f);
	
	String* s = str_create(content, r);
	return s;
}

/*
 * A lazy programer's function that prints the content of a String structure,
 * then a '\n' and then flushes the stdout stream.
 */
void str_println(String* str) {
	printf(str->value);
	printf("\n");
	fflush (stdout);
}

/*
 * This method safely deallocate the GridInfo structure.
 */
void grid_info_free(GridInfo* info) {
	if (info) {
		
		if (info->max)
			free(info->max);
		
		if (info->min)
			free(info->min);
		
		if (info->mean)
			free(info->mean);
		
		if (info->stdev)
			free(info->stdev);
		
		if (info->discrete)
			free(info->discrete);
		
		if (info->numbers)
			free(info->numbers);
		
		if (info->words)
			free(info->words);
		
		if (info->missing)
			free(info->missing);
		
		free(info);
	}
}

/*
 * This method safely deallocate the Grid structure and some of it's internal
 * structures. The number of rows and columns to be deallocated is given by
 * the rows and cols parameters respectively.
 */
void grid_partial_free(Grid* g, size_t rows, size_t cols) {
	if (g) {
		
		char*** body = g->body;
		if (body) {
			
			for (size_t i = 0; i < rows; i++) {
				if (body[i])
					for (size_t j = 0; j < cols; j++) {
						
						if (body[i][j]) {
							free(body[i][j]);
						}
						
					}
				free(body[i]);
			}
			
			free(body);
		}
		
		grid_info_free(g->info);
		
		free(g);
	}
}

/*
 * Safely deallocates the Grid structure.
 */
void grid_free(Grid* g) {
	if (g) {
		
		char*** body = g->body;
		if (body) {
			if (g->info) {
				size_t m = g->info->rows;
				size_t n = g->info->columns;
				for (size_t i = 0; i < m; i++) {
					if (body[i]) {
						for (size_t j = 0; j < n; j++) {
							if (body[i][j]) {
								free(body[i][j]);
							}
						}
						free(body[i]);
						
					}
				}
			}
			free(body);
			
		}
		
		grid_info_free(g->info);
		free(g);
	}
}

/*
 * Dealocates the given String structure.
 */
void str_free(String* str) {
	free((char*) str->value);
	free(str);
}

/*
 * An utility method that creates a char array of l+1 length and
 * assigns '\0' to the last position.
 */
char* char_create_arr(size_t l) {
	char* s = malloc(sizeof(char) * (l + 1));
	s[l] = 0;
	return s;
}

/*
 * Creates a String structure with the given value pointer and length parameters.
 */
String* str_create(char* value, size_t length) {
	String* s = malloc(sizeof(String));
	s->length = length;
	s->value = value;
	return s;
}

/*
 * Counts how many characters (denoted by the given c parameter) are
 * in the given str string.
 */
size_t char_count(char* str, size_t end, char c) {
	if (end == -1) {
		end = LONG_MAX;
	}
	
	size_t sum = 0;
	char crt;
	for (size_t i = 0; i < end; i++) {
		crt = str[i];
		if (crt == 0) {
			//we reached the end, no need to keep going.
			return sum;
		}
		
		if (crt == c) {
			sum++;
		}
	}
	
	return sum;
}

/*
 * This method scan thru all the values of a given column in a Grid structure
 * and returns those that are different from each other. In the case a column
 * has the same value for all the rows, then a single value will be returned.
 */
char** str_uniq(Grid* g, size_t col, size_t* destLength) {
	
	char*** body = g->body;
	size_t m = g->info->rows;
	
	char** temp = malloc(sizeof(char*) * m);
	
	size_t assigned = 0;
	for (int i = 0; i < m; i++) {
		char* crt = body[i][col];
		
		//let's search in the temp array to see if we already found it
		short found = 0;
		for (int j = 0; j < assigned; j++) {
			char* other = temp[j];
			if (crt == NULL && other == NULL) {
				found = 1;
				break;
			}
			
			if (crt == NULL || other == NULL) {
				continue;
			}
			
			if (strcmp(crt, other) == 0) {
				found = 1;
				break;
			}
		}
		
		if (!found) {
			//we support NULL values, so let us check first
			if (crt) {
				size_t l = strlen(crt);
				char* copy = char_create_arr(l);
				strcpy(copy, crt);
				temp[assigned] = copy;
				assigned++;
			} else {
				temp[assigned] = NULL;
				assigned++;
			}
		}
	}
	
	char** uniq = malloc(sizeof(char*) * assigned);
	
	for (int i = 0; i < assigned; i++) {
		uniq[i] = temp[i];
	}
	
	free(temp);
	*destLength = assigned;
	return uniq;
}

/*
 * This method computes the length of a given char array if it would be trimmed
 * (removing leading and trailing white spaces).
 */
size_t trim_lenght(char* src, size_t length) {
	ssize_t right = char_not_find(src, ' ');
	if (right == -1) {
		return 0;
	}
	
	ssize_t left = char_not_find_backwards(src, length, ' ');
	if (left == -1) {
		return 0;
	}
	
	return left - right + 1;
}

/*
 * Copies a trimmed (without leading and trailing white spaces) version of a
 * given string to a destination char pointer.
 */
size_t trim(char* src, size_t srcLength, char* dest, size_t destLength) {
	
	ssize_t right = char_not_find(src, ' ');
	if (right == -1) {
		return 0;
	}
	
	size_t left = char_not_find_backwards(src, srcLength, ' ');
	if (left == -1) {
		return 0;
	}
	
	left++;
	
	size_t j = 0;
	for (size_t i = right; i < left; i++) {
		dest[j] = src[i];
		j++;
	}
	
	return j;
}

/*
 * Returns the value of a if a > b, and b if b > a. Note that for this method
 * any value is higher than NAN.
 */
float higher(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a > b) {
		return a;
	}
	
	return b;
}

/*
 * Returns the value of a if a < b, and b if b < a. Note that for this method
 * any value is lower than NAN.
 */
float lower(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a < b) {
		return a;
	}
	
	return b;
}

/*
 * Creates a m x n Matrix struct filled with zeros.
 */
Matrix* mtr_create(size_t m, size_t n) {
	
	Matrix* matrix = malloc(sizeof(Matrix));
	matrix->m = m;
	matrix->n = n;
	matrix->values = calloc(m * n, sizeof(float));
	
	return matrix;
}

/*
 * Creates a m x n Matrix struct using the given values parameters as the
 * matrix's values. Note that this method won't copy the values given, it
 * just use it as a reference pointer.
 */
Matrix* mtr_create_filled(size_t m, size_t n, float* values) {
	
	Matrix* matrix = malloc(sizeof(Matrix));
	matrix->m = m;
	matrix->n = n;
	matrix->values = values;
	
	return matrix;
}

/*
 * Safely deallocate the given Matrix struct.
 */
void mtr_free(Matrix* matrix) {
	if (matrix) {
		if (matrix->values) {
			free(matrix->values);
		}
		
		free(matrix);
	}
}

/*
 * This method selects by column the elements of a given Matrix struct and
 * stores it in a new Matrix. The returned matrix dimension is
 * m x (fromIdx-toIdx) where m is the number of row of the given matrix.
 */
Matrix* mtr_col_slct(Matrix* mtr, size_t fromIdx, size_t toIdx) {
	if (toIdx <= fromIdx || fromIdx < 0) {
		fflush (stdout);
		fprintf(stderr, "Invalid arguments. startInc: %d, endExc: %d.", fromIdx, toIdx);
		fflush (stderr);
		return NULL;
	}
	
	size_t m = mtr->m;
	size_t n = toIdx - fromIdx;
	
	float* src = mtr->values;
	size_t srcn = mtr->n;
	
	Matrix* matrix = mtr_create(m, n);
	float* values = matrix->values;
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			values[i * n + j] = src[i * srcn + fromIdx + j];
		}
	}
	
	return matrix;
}

/*
 * This method selects by row the elements of a given Matrix struct and
 * stores it in a new Matrix. The returned matrix dimension is
 * (toIdx-fromIdx) x n where n is the number of columns of the given matrix.
 */
Matrix* mtr_row_slct(Matrix* mtr, size_t fromIdx, size_t toIdx) {
	if (toIdx <= fromIdx || fromIdx < 0) {
		fflush (stdout);
		fprintf(stderr, "Invalid arguments. fromIdx: %d, toIdx: %d.\n", fromIdx, toIdx);
		fflush (stderr);
		return NULL;
	}
	
	size_t m = toIdx - fromIdx;
	size_t n = mtr->n;
	
	float* src = mtr->values;
	
	Matrix* matrix = mtr_create(m, n);
	float* values = matrix->values;
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			values[i * n + j] = src[(fromIdx + i) * n + j];
		}
	}
	
	return matrix;
}

/*
 * This methods maps a given value from a Grid internal values
 * (given by the map parameter) to a column in the Grid's Matrix. Since the
 * number of columns in the matrix can be higher than the number of columns
 * in the Grid, it is important to use this method for the column mapping
 * from Grid to Matrix.
 */
size_t to_mtr_col(char*** map, size_t* sizes, size_t* missing, size_t col, char* val) {
	size_t left = mtr_cols(map, sizes, missing, col);
	if (map[col] == NULL) {
		//easy case when there are no missing, just return right what mtr_cols returned.
		if (missing[col] == 0) {
			return left;
		}
		
		if (val == NULL) {
			return left;
		}
		
		return left + 1;
	}
	
	size_t l = sizes[col];
	for (size_t i = 0; i < l; i++) {
		char* crt = map[col][i];
		
		if (crt == NULL && val == NULL) {
			return left + i;
		}
		
		if ((crt != NULL && val != NULL) && strcmp(crt, val) == 0) {
			return left + i;
		}
	}
	
	fprintf(stderr, "could not convert to matrix "
			"column for grid column %d and value %s", col, val);
	fflush (stderr);
	return -1;
}

/*
 * Shuffles the rows of a given matrix.
 */
void mtr_shuffle(Matrix* matrix) {
	
	float* values = matrix->values;
	size_t m = matrix->m;
	size_t n = matrix->n;
	
	float* buffer = malloc(sizeof(float) * n);
	for (size_t i = 0; i < m; i++) {
		size_t idxTo = (size_t)((m - 1) * (rand() / (double) RAND_MAX));
		mtr_swap(values, n, buffer, i, idxTo);
	}
	free(buffer);
}

/*
 * Swaps two rows.
 */
void mtr_swap(float* values, size_t n, float* buffer, size_t idxFrom, size_t idxTo) {
	if (idxFrom == idxTo) {
		return;
	}
	
	//first copies the row at idxTo row in to a buffer.
	for (size_t j = 0; j < n; j++) {
		buffer[j] = values[idxTo * n + j];
	}
	
	//then writes the values of the row at idxFrom to the row at idxTo
	for (size_t j = 0; j < n; j++) {
		values[idxTo * n + j] = values[idxFrom * n + j];
	}
	
	//and last, writes the values of the buffer to the row at idxFrom
	for (size_t j = 0; j < n; j++) {
		values[idxFrom * n + j] = buffer[j];
	}
}

/*
 * Computes the standard deviation and the mean for each column in the grid
 * and stores it in the given GridInfo struct.
 */
void fill_stdev(GridInfo* info, Grid* g) {
	
	size_t cols = info->columns;
	float* means = info->mean;
	float* stdevs = info->stdev;
	
	for (size_t i = 0; i < cols; i++) {
		float mean;
		float stdev;
		online_stdev(info, g, i, &mean, &stdev);
		means[i] = mean;
		stdevs[i] = stdev;
	}
}

/*
 * This method computes the standard deviation and the mean using an online algorithm as described in here:
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm.
 */
void online_stdev(GridInfo* info, Grid* g, size_t col, float* mean, float* stdev) {
	
	size_t m = info->rows;
	if (m < 2 || info->words[col] > 0) {
		//there is no sense on computing the standard deviation or the
		//mean with only 1 instance.
		*mean = NAN;
		*stdev = NAN;
		return;
	}
	
	char*** body = g->body;
	
	double n = 0;
	double _mean = 0.0;
	double m2 = 0.0;
	for (size_t i = 0; i < m; i++) {
		char* val = body[i][col];
		
		if (val == NULL) {
			fprintf(stderr, "The value in grid at row: %d and column %d is null or empty.\n", i,
					col);
			fflush (stderr);
			exit (EXIT_FAILURE);
		}
		
		char* tailptr;
		double x = strtod(val, &tailptr);
		if (val == tailptr) {
			//not a number
		} else {
			
			n++;
			
			double delta = x - _mean;
			_mean += delta / n;
			m2 += delta * (x - _mean);
		}
		
	}
	
	*mean = (float) _mean;
	*stdev = (float) m2 / (n - 1);
}

/*
 * Nicely prints the values of a Grid struct.
 */
void grid_print(Grid* g) {
	size_t m = g->info->rows;
	size_t n = g->info->columns;
	char*** values = g->body;
	
	for (size_t i = 0; i < m; i++) {
		printf("%5d  ", (i + 1));
		for (size_t j = 0; j < n; j++) {
			printf("%5s  ", values[i][j]);
		}
		printf("\n");
	}
}

/*
 * Nicely prints the values of a Matrix.
 */
void mtr_print(Matrix* matrix) {
	int m = matrix->m;
	int n = matrix->n;
	float* values = matrix->values;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%7.4f", values[i * n + j]);
			if (j + 1 < n) {
				printf("  ");
			}
		}
		printf("\n");
	}
	printf("\n");
	fflush (stdout);
}


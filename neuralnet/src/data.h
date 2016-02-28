/*
 * data.h
 *
 *  Created on: Jan 25, 2016
 *      Author: yaison
 */

#ifndef DATA_H_
#define DATA_H_

#include <sys/types.h>

typedef struct String {
		size_t length;
		char* value;
} String;

typedef struct GridInfo {
		float* max;
		float* min;
		float* mean;
		float* stdev;

		short* discrete;
		size_t* numbers;
		size_t* words;
		size_t* missing;

		size_t rows;
		size_t columns;
} GridInfo;

typedef struct Grid {
		
		GridInfo* info;
		char*** body;
} Grid;

typedef struct Mapper {
		size_t cols;
		size_t* sizes;
		char*** map;
} Mapper;

typedef struct Matrix {
		size_t m;
		size_t n;
		float* values;
} Matrix;

unsigned int lowercmp(char* str, char* other);
void mtrshuffle(Matrix* matrix);
void mtrswap(float* values, size_t n, float* buffer, size_t idxFrom, size_t idxTo);
Matrix* mtr_create_grid(Grid* g, Mapper* mapper, int scale);
Matrix* mtr_create_filled(size_t m, size_t n, float* values);
size_t mtr_cols(char*** map, size_t* sizes, size_t* missing, size_t cols);
Matrix* mtrrange(Matrix* mtr, size_t fromIdx, size_t toIdx);
Matrix* mtr_col_slct(Matrix* mtr, size_t startInc, size_t endExc);
Matrix* mtr_row_slct(Matrix* mtr, size_t fromIdx, size_t toIdx);
void mtr_shuffle(Matrix* matrix);
void mtr_swap(float* values, size_t n, float* buffer, size_t idxFrom, size_t idxTo);
Matrix* mtrxcl(Matrix* mtr, size_t col);
Matrix* mtr_create(size_t m, size_t n);
void mtr_print(Matrix* matrix);
void mtr_free(Matrix* matrix);

void fill_stdev(GridInfo* info, Grid* g);
void online_stdev(GridInfo* info, Grid* g, size_t col, float* mean, float* stdev);

void grid_print(Grid* g);
size_t to_mtr_col(char*** map, size_t* sizes, size_t* missing, size_t col, char* val);

Mapper* map_create(Grid* g);
void map_free(Mapper* mapper);
char** str_uniq(Grid* g, size_t col, size_t* destLength);
size_t trim_lenght(char* src, size_t srcLength);
size_t trim(char* src, size_t srcLength, char* dest, size_t destLength);

float higher(float a, float b);
float lower(float a, float b);

GridInfo* grid_info(Grid* g, size_t rows, size_t cols);
void grid_info_free(GridInfo* info);
Grid* grid_create(char* raw, char d);

void grid_free(Grid* g);
void grid_partial_free(Grid* g, size_t rows, size_t cols);

size_t char_count(char* str, size_t end, char c);
void flush();
ssize_t char_find(char* str, char c);
ssize_t char_not_find(char* str, char c);
ssize_t char_not_find_backwards(char* str, size_t length, char c);
String* ffull(char* path);
void str_println(String* str);
char* char_create_arr(size_t l);

void str_free(String* str);
String* str_create(char* value, size_t length);

#endif /* DATA_H_ */

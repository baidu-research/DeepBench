/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdint>
#include <sstream>

#include <mkl.h>
#include "gemm_problems.h"

#define FIX_LD(x)   (((((x) + 127)/128)*128) + 16)

#define REPEAT 10
#define MKL_MEM_ALIGNMENT (4*1024)

#ifdef IGEMM_S8U8S32
#define A_TYPE MKL_INT8
#define B_TYPE MKL_UINT8
#define C_TYPE MKL_INT32
#ifdef PACKED_API
#error "packed api is not supported for integer GEMM"
#endif
#else
#define A_TYPE float
#define B_TYPE float
#define C_TYPE float
#endif

typedef struct gemm_params {
    bool ta;
    bool tb;
    char transa;
    char transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
} gemm_params_t;



int main(int argc, char *argv[])
{
  int i, j;
  size_t sizea, sizeb, sizec;
  size_t num_gemms;
  size_t max_sizea = 0, max_sizeb = 0, max_sizec = 0;
  int max_m = 0, max_n = 0, max_k = 0;
  A_TYPE  *A, ao = 0, bo = 0;
  B_TYPE  *B;
  C_TYPE  *C, co = 0;
  float  alpha = 1.0, beta = 1.0;
  double flops, total_flops = 0., st_time, end_time, ave_time, total_time = 0.;
#ifdef PACKED_API
  float *AP, *BP;
#endif

  int run_training_set = 1;
  if (argc > 1) run_training_set = atoi(argv[1]);

  std::vector<std::tuple<int, int, int, bool, bool>>* p_problem_set;
  if (run_training_set) {
    printf("Running the training benchmark (set first program argument to 0 for inference)\n");
    p_problem_set = &training_set;
  } else {
    printf("Running the inference benchmark (first program argument is 0)\n");
    p_problem_set = &inference_server_set;
  }

  num_gemms = p_problem_set->size();
  gemm_params_t* p_gemm_params = (gemm_params_t*) _mm_malloc(num_gemms*sizeof(gemm_params_t), 64);

  i = 0;
  for (const auto &problem : *p_problem_set) {
    std::tie(p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k, 
             p_gemm_params[i].ta, p_gemm_params[i].tb) = problem;

    if (p_gemm_params[i].ta) {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].k);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].m;
      p_gemm_params[i].transa = 'T';
    } else {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].m);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].k;
      p_gemm_params[i].transa = 'N';
    }

    if (p_gemm_params[i].tb) {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].n);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].k;
      p_gemm_params[i].transb = 'T';
    } else {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].k);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].n;
      p_gemm_params[i].transb = 'N';
    }

    p_gemm_params[i].ldc = FIX_LD(p_gemm_params[i].m);
    sizec = p_gemm_params[i].ldc * p_gemm_params[i].n;

    max_sizea = std::max(sizea, max_sizea);
    max_sizeb = std::max(sizeb, max_sizeb);
    max_sizec = std::max(sizec, max_sizec);

    max_m     = std::max(p_gemm_params[i].m, max_m);
    max_n     = std::max(p_gemm_params[i].n, max_n);
    max_k     = std::max(p_gemm_params[i].k, max_k);
    ++i;
  }

  assert(i == num_gemms);

  A = (A_TYPE*) mkl_malloc(sizeof(A_TYPE)*max_sizea, MKL_MEM_ALIGNMENT);
  B = (B_TYPE*) mkl_malloc(sizeof(B_TYPE)*max_sizeb, MKL_MEM_ALIGNMENT);
  C = (C_TYPE*) mkl_malloc(sizeof(C_TYPE)*max_sizec, MKL_MEM_ALIGNMENT);
#ifdef PACKED_API
  AP = sgemm_alloc("A", &max_m, &max_n, &max_k);
  BP = sgemm_alloc("B", &max_m, &max_n, &max_k);
#endif

#ifdef IGEMM_S8U8S32
  for (i=0; i<max_sizea; ++i) A[i] = 11;
  for (i=0; i<max_sizeb; ++i) B[i] = 22;
  for (i=0; i<max_sizec; ++i) C[i] = 33;
#else
  for (i=0; i<max_sizea; ++i) A[i] = (float) drand48();
  for (i=0; i<max_sizeb; ++i) B[i] = (float) drand48();
  for (i=0; i<max_sizec; ++i) C[i] = (float) drand48();
#endif

  for (i=0; i < num_gemms; ++i) {

#ifdef PACKED_API
    sgemm_pack("A", &p_gemm_params[i].transa, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, &alpha, A, &p_gemm_params[i].lda, AP);
    sgemm_pack("B", &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, &alpha, B, &p_gemm_params[i].ldb, BP);
    // warmup
    sgemm_compute("P", "P", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
            AP, &p_gemm_params[i].lda, BP, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
    st_time = dsecnd();
    for (j = 0; j < REPEAT; ++j) {
      sgemm_compute("P", "P", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          AP, &p_gemm_params[i].lda, BP, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
    }
#else
    // warmup
#ifdef IGEMM_S8U8S32
    gemm_s8u8s32(&p_gemm_params[i].transa, &p_gemm_params[i].transb, "F", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
        &alpha, A, &p_gemm_params[i].lda, &ao, B, &p_gemm_params[i].ldb, &bo, &beta, C, &p_gemm_params[i].ldc, &co);
#else
    sgemm(&p_gemm_params[i].transa, &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
        &alpha, A, &p_gemm_params[i].lda, B, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
#endif
    // time measurements
    st_time = dsecnd();
    for (j = 0; j < REPEAT; ++j) {
#ifdef IGEMM_S8U8S32
      gemm_s8u8s32(&p_gemm_params[i].transa, &p_gemm_params[i].transb, "F", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          &alpha, A, &p_gemm_params[i].lda, &ao, B, &p_gemm_params[i].ldb, &bo, &beta, C, &p_gemm_params[i].ldc, &co);
#else
      sgemm(&p_gemm_params[i].transa, &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          &alpha, A, &p_gemm_params[i].lda, B, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
#endif
    }
#endif
    end_time = dsecnd();

    flops = 2.*p_gemm_params[i].m*p_gemm_params[i].n*p_gemm_params[i].k;
    total_flops += flops;

    ave_time     = 1E6*(end_time - st_time)/REPEAT;
    total_time  += ave_time;

#ifdef IGEMM_S8U8S32
    printf("GEMM_S8U8S32(%c,%c,%d,%d,%d) %.1f usec %.5f GOp/sec \n",
        p_gemm_params[i].transa, p_gemm_params[i].transb, 
        p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k,
        ave_time, 1E-3*flops/ave_time);
#else
    printf("SGEMM(%c,%c,%d,%d,%d) %.1f usec %.5f GFlop/sec \n",
        p_gemm_params[i].transa, p_gemm_params[i].transb, 
        p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k,
        ave_time, 1E-3*flops/ave_time);
#endif
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
#ifdef PACKED_API
  sgemm_free(AP);
  sgemm_free(BP);
#endif


#ifdef IGEMM_S8U8S32
  printf("Total time %.1f usec, Overall Performance: %.5f GOp/sec \n", total_time, 1E-3*total_flops/total_time);
#else
  printf("Total time %.1f usec, Overall Performance: %.5f GFlop/sec \n", total_time, 1E-3*total_flops/total_time);
#endif
  return 0;
}

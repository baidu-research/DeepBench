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
#include "mkl.h"
#include "input.h"

#define IS_TRANS(x) (((x) == 'T') || ((x) == 't'))
#define FIX_LD(x)   (((((x) + 127)/128)*128) + 16)
#define MAX(x, y)   ((x) >= (y) ? (x) : (y))

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
  gemm_params_t* p_gemm_params;

  int run_training_set = 1;
  if (argc > 1) run_training_set = atoi(argv[1]);


  if (run_training_set) {
    printf("Running the training benchmark (set first program argument to 0 for inference)\n");
    p_gemm_params = &gemm_training_params[0];
    num_gemms = sizeof(gemm_training_params)/sizeof(gemm_training_params[0]);
  } else {
    printf("Running the inference benchmark (first program argument is 0)\n");
    p_gemm_params = &gemm_server_inference_params[0];
    num_gemms = sizeof(gemm_server_inference_params)/sizeof(gemm_server_inference_params[0]);
  }

  for (i=0; i < num_gemms; ++i) {

    if (IS_TRANS(p_gemm_params[i].transa)) {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].k);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].m;
    } else {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].m);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].k;
    }

    if (IS_TRANS(p_gemm_params[i].transb)) {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].n);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].k;
    } else {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].k);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].n;
    }

    p_gemm_params[i].ldc = FIX_LD(p_gemm_params[i]. m);
    sizec = p_gemm_params[i].ldc * p_gemm_params[i].n;

    max_sizea = MAX(sizea, max_sizea);
    max_sizeb = MAX(sizea, max_sizeb);
    max_sizec = MAX(sizec, max_sizec);

    max_m     = MAX(p_gemm_params[i].m, max_m);
    max_n     = MAX(p_gemm_params[i].n, max_n);
    max_k     = MAX(p_gemm_params[i].k, max_k);
  }

  A = mkl_malloc(sizeof(A_TYPE)*max_sizea, MKL_MEM_ALIGNMENT);
  B = mkl_malloc(sizeof(B_TYPE)*max_sizeb, MKL_MEM_ALIGNMENT);
  C = mkl_malloc(sizeof(C_TYPE)*max_sizec, MKL_MEM_ALIGNMENT);
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

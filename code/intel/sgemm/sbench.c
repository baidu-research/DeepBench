/*******************************************************************************
* Copyright 2016 Intel Corporation
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
#define FIX_LD(x)   (((((x) + 127)/128)*128) + 64)
#define MAX(x, y)   ((x) >= (y) ? (x) : (y))

#define REPEAT 10
#define MKL_MEM_ALIGNMENT (2*1024)

int main(int argc, char *argv[])
{
  int i, j;
  size_t sizea, sizeb, sizec;
  size_t max_sizea = 0, max_sizeb = 0, max_sizec = 0;
  int max_m = 0, max_n = 0, max_k = 0;
  float  *A, *B, *C;
  float  alpha = 1.0, beta = 1.0;
  double flops, total_flops = 0., st_time, end_time, ave_time, total_time = 0.;
#ifdef PACKED_API
  float *AP, *BP;
#endif

  for (i=0; i < sizeof(gemm_params)/sizeof(gemm_params[0]); ++i) {

    if (IS_TRANS(gemm_params[i].transa)) {
      gemm_params[i].lda = FIX_LD(gemm_params[i].k);
      sizea = gemm_params[i].lda * gemm_params[i].m;
    } else {
      gemm_params[i].lda = FIX_LD(gemm_params[i].m);
      sizea = gemm_params[i].lda * gemm_params[i].k;
    }

    if (IS_TRANS(gemm_params[i].transb)) {
      gemm_params[i].ldb = FIX_LD(gemm_params[i].n);
      sizeb = gemm_params[i].ldb * gemm_params[i].k;
    } else {
      gemm_params[i].ldb = FIX_LD(gemm_params[i].k);
      sizeb = gemm_params[i].ldb * gemm_params[i].n;
    }

    gemm_params[i].ldc = FIX_LD(gemm_params[i]. m);
    sizec = gemm_params[i].ldc * gemm_params[i].n;

    max_sizea = MAX(sizea, max_sizea);
    max_sizeb = MAX(sizea, max_sizeb);
    max_sizec = MAX(sizec, max_sizec);

    max_m     = MAX(gemm_params[i].m, max_m);
    max_n     = MAX(gemm_params[i].n, max_n);
    max_k     = MAX(gemm_params[i].k, max_k);
  }

  A = mkl_malloc(sizeof(float)*max_sizea, MKL_MEM_ALIGNMENT);
  B = mkl_malloc(sizeof(float)*max_sizeb, MKL_MEM_ALIGNMENT);
  C = mkl_malloc(sizeof(float)*max_sizec, MKL_MEM_ALIGNMENT);
#ifdef PACKED_API
  AP = sgemm_alloc("A", &max_m, &max_n, &max_k);
  BP = sgemm_alloc("B", &max_m, &max_n, &max_k);
#endif

  for (i=0; i<max_sizea; ++i) A[i] = (float) drand48();
  for (i=0; i<max_sizeb; ++i) B[i] = (float) drand48();
  for (i=0; i<max_sizec; ++i) C[i] = (float) drand48();

  for (i=0; i < sizeof(gemm_params)/sizeof(gemm_params[0]); ++i) {
    // warmup
    sgemm(&gemm_params[i].transa, &gemm_params[i].transb, &gemm_params[i].m, &gemm_params[i].n, &gemm_params[i].k, 
        &alpha, A, &gemm_params[i].lda, B, &gemm_params[i].ldb, &beta, C, &gemm_params[i].ldc);

    // time measurements
    st_time = dsecnd();
#ifdef PACKED_API
    sgemm_pack("A", &gemm_params[i].transa, &gemm_params[i].m, &gemm_params[i].n, &gemm_params[i].k, &alpha, A, &gemm_params[i].lda, AP);
    sgemm_pack("B", &gemm_params[i].transb, &gemm_params[i].m, &gemm_params[i].n, &gemm_params[i].k, &alpha, B, &gemm_params[i].ldb, BP);
    st_time = dsecnd();
    for (j = 0; j < REPEAT; ++j) {
      sgemm_compute("P", "P", &gemm_params[i].m, &gemm_params[i].n, &gemm_params[i].k, 
          AP, &gemm_params[i].lda, BP, &gemm_params[i].ldb, &beta, C, &gemm_params[i].ldc);
    }
#else
    for (j = 0; j < REPEAT; ++j) {
      sgemm(&gemm_params[i].transa, &gemm_params[i].transb, &gemm_params[i].m, &gemm_params[i].n, &gemm_params[i].k, 
          &alpha, A, &gemm_params[i].lda, B, &gemm_params[i].ldb, &beta, C, &gemm_params[i].ldc);
    }
#endif
    end_time = dsecnd();

    flops = 2.*gemm_params[i].m*gemm_params[i].n*gemm_params[i].k;
    total_flops += flops;

    ave_time     = 1E6*(end_time - st_time)/REPEAT;
    total_time  += ave_time;

    printf("SGEMM(%c,%c,%d,%d,%d) %.1f usec %.5f GFlop/sec \n",
        gemm_params[i].transa, gemm_params[i].transb, 
        gemm_params[i].m, gemm_params[i].n, gemm_params[i].k,
        ave_time, 1E-3*flops/ave_time);
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
#ifdef PACKED_API
  sgemm_free(AP);
  sgemm_free(BP);
#endif

  printf("Total time %.1f usec, Overall Performance: %.5f GFlop/sec \n", total_time, 1E-3*total_flops/total_time);
  return 0;
}

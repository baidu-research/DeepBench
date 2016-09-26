/*******************************************************************************
 * Copyright 2016 Intel Corporation All Rights Reserved.
 *
 * The source code,  information  and material  ("Material") contained  herein is
 * owned by Intel Corporation or its  suppliers or licensors,  and  title to such
 * Material remains with Intel  Corporation or its  suppliers or  licensors.  The
 * Material  contains  proprietary  information  of  Intel or  its suppliers  and
 * licensors.  The Material is protected by  worldwide copyright  laws and treaty
 * provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
 * modified, published,  uploaded, posted, transmitted,  distributed or disclosed
 * in any way without Intel's prior express written permission.  No license under
 * any patent,  copyright or other  intellectual property rights  in the Material
 * is granted to  or  conferred  upon  you,  either   expressly,  by implication,
 * inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Unless otherwise agreed by Intel in writing,  you may not remove or alter this
 * notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
 * suppliers or licensors in any way.
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

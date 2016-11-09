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


#define _GNU_SOURCES
#define _GNU_SOURCES
#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <assert.h>
#include <immintrin.h>

#include <omp.h>

#include "mkl_dnn.h"
#include "input.h"

#define FWD_CONVOLUTION   1
#define BWD_F_CONVOLUTION 2
#define BWD_D_CONVOLUTION 3

#define PAD 1
#define NO_PAD 0

#define dimension (4)

#define MAX2(x,y) (((x)>(y))?(x):(y))
#define MIN2(x,y) (((x)<(y))?(x):(y))

#define DEFAULT_STRIDES(strides, size, count) do { \
    ((strides)[0]) = 1; \
    for (size_t i = 1; i < count; ++i) \
        ((strides)[i]) = ((strides)[i-1])*((size)[i-1]); \
} while(0);

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
        ret = -1; \
        goto bail_out; \
    } \
} while(0);

static inline long long int getWallClockNanos() {
    struct timespec tv;
    int errcode;

    errcode = clock_gettime(CLOCK_MONOTONIC,&tv);
    assert(errcode == 0);

    return 1000000000ll * tv.tv_sec + tv.tv_nsec;
}

static inline double getWallClockSeconds() {
    return getWallClockNanos() / 1e9;
}

static int bench_conv(conv_params_t param, int mode, int pad)
{
    struct timespec start;
    struct timespec  stop;

    dnnError_t err;
    int ret = 0;
    size_t batch_size = 16;

    size_t outputSize[dimension], outputStride[dimension];
    size_t inputSize[dimension], inputStride[dimension];
    size_t filterSize[dimension+1], filterStride[dimension+1];
    size_t biasSize[1], biasStride[1];

    size_t convolutionStride[dimension - 2];
    int inputOffset[dimension - 2 ];

    int groups       = param.groups;
    size_t minibatch = param.minibatch;

    inputSize[0] = param.w;
    inputSize[1] = param.h;
    inputSize[2] = param.ic;
    inputSize[3] = param.minibatch;
    DEFAULT_STRIDES(inputStride , inputSize , 4);

    filterSize[0] = param.kw;
    filterSize[1] = param.kh;
    filterSize[2] = param.ic / MAX2(1,groups);
    filterSize[3] = param.oc / MAX2(1,groups);
    filterSize[4] = groups;
    DEFAULT_STRIDES(filterStride, filterSize, 5);

    convolutionStride[0] = param.c_stride;
    convolutionStride[1] = param.c_stride;

    inputOffset[0] = -param.offset;
    inputOffset[1] = -param.offset;

    outputSize[0] = (inputSize[0] - filterSize[0] + 2*(-inputOffset[0])) / convolutionStride[0] + 1; // W
    outputSize[1] = (inputSize[1] - filterSize[1] + 2*(-inputOffset[1])) / convolutionStride[1] + 1; // H
    outputSize[2] = param.oc;
    outputSize[3] = minibatch;
    DEFAULT_STRIDES(outputStride, outputSize , 4);

    biasSize[0] = outputSize[2];
    biasStride[0] = 1;

    int NTIMES = param.iters;
    if (NTIMES > 1024) {
      ret = -1;
      goto bail_out;
    }
    srand48(1);

    double flops = 0.0;
    if (pad == PAD) // Padding IS include into flops
        flops =(2.0*(double)minibatch*
                    (double)filterSize[0]*(double)filterSize[1]*  // KW * KH
                    (double)outputSize[0]*(double)outputSize[1]*  // OW * OH
                    (double)inputSize[2] *(double)filterSize[3]); // IC * OC
    else if (pad == NO_PAD) { // Padding IS NOT include into flops
        long long ops = 0;
        auto in = [](int x, int l, int r) -> bool { return (l <= x && x < r); };
        // XXX: double check me
        for (int yptr = 0; yptr + filterSize[1] <= inputSize[1] + 2*(-inputOffset[1]); yptr += convolutionStride[1]) {
          for (int xptr = 0; xptr + filterSize[0] <= inputSize[0] + 2*(-inputOffset[0]); xptr += convolutionStride[0]) {
            for (int dy = 0; dy < filterSize[1]; ++dy)
              for (int dx = 0; dx < filterSize[0]; ++dx)
                ops += in(yptr+dy, (-inputOffset[1]),inputSize[1]+(-inputOffset[1]))
                    && in(xptr+dx, (-inputOffset[0]),inputSize[0]+(-inputOffset[0]));
          }
        }
        ops *= 2*minibatch*inputSize[2]*filterSize[3];
        flops = (double) ops;
    } else {
        ret = -1;
        goto bail_out;
    }

    dnnPrimitive_t conv = NULL;
    dnnLayout_t lt_conv_src         = NULL,
                lt_conv_diff_src    = NULL,
                lt_conv_dst         = NULL,
                lt_conv_diff_dst    = NULL,
                lt_conv_filter      = NULL,
                lt_conv_diff_filter = NULL,
                lt_conv_bias        = NULL;

    float* resconv[dnnResourceNumber] = {0};

    /*** convolution section ***/
    if (mode == FWD_CONVOLUTION) {
        CHECK_ERR( dnnGroupsConvolutionCreateForwardBias_F32(&conv, NULL,
                   dnnAlgorithmConvolutionDirect, groups, dimension, inputSize,
                   outputSize, filterSize, convolutionStride, inputOffset,
                   dnnBorderZeros), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_dst,conv, dnnResourceDst) , err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_src, conv, dnnResourceSrc) , err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_filter, conv, dnnResourceFilter), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_bias, conv, dnnResourceBias) , err );

        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDst], lt_conv_dst), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceSrc],lt_conv_src), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceFilter],lt_conv_filter), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceBias],lt_conv_bias), err );

        size_t lt_conv_dst_len = dnnLayoutGetMemorySize_F32(lt_conv_dst) / sizeof(float);
        size_t lt_conv_src_len = dnnLayoutGetMemorySize_F32(lt_conv_src) / sizeof(float);
        size_t lt_conv_filter_len = dnnLayoutGetMemorySize_F32(lt_conv_filter) / sizeof(float);
        size_t lt_conv_bias_len = dnnLayoutGetMemorySize_F32(lt_conv_bias) / sizeof(float);

        for (size_t i = 0; i < lt_conv_dst_len; i++) resconv[dnnResourceDst][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_src_len; i++) resconv[dnnResourceSrc][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_filter_len; i++) resconv[dnnResourceFilter][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_bias_len; i++) resconv[dnnResourceBias][i] = (float)drand48();
    } else if (mode == BWD_D_CONVOLUTION) {
        CHECK_ERR( dnnConvolutionCreateBackwardData_F32 (&conv, NULL,
                    dnnAlgorithmConvolutionDirect, dimension, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros), err);
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_src, conv, dnnResourceDiffSrc), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_dst, conv, dnnResourceDiffDst), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_filter, conv, dnnResourceFilter), err );

        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffSrc],lt_conv_diff_src), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffDst],lt_conv_diff_dst), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceFilter],lt_conv_filter), err );


        size_t lt_conv_diff_src_len = dnnLayoutGetMemorySize_F32(lt_conv_diff_src) / sizeof(float);
        size_t lt_conv_diff_dst_len = dnnLayoutGetMemorySize_F32(lt_conv_diff_dst) / sizeof(float);
        size_t lt_conv_filter_len = dnnLayoutGetMemorySize_F32(lt_conv_filter) / sizeof(float);

        for (size_t i = 0; i < lt_conv_diff_src_len; i++) resconv[dnnResourceDiffSrc][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_diff_dst_len; i++) resconv[dnnResourceDiffDst][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_filter_len; i++) resconv[dnnResourceFilter][i] = (float)drand48();
    } else if (mode == BWD_F_CONVOLUTION) {
        CHECK_ERR( dnnConvolutionCreateBackwardFilter_F32 (&conv, NULL,
                    dnnAlgorithmConvolutionDirect, dimension, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros), err);
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_filter, conv, dnnResourceDiffFilter), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_diff_dst, conv, dnnResourceDiffDst), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv_src, conv, dnnResourceSrc), err );

        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffFilter], lt_conv_diff_filter), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceDiffDst],lt_conv_diff_dst), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&resconv[dnnResourceSrc],lt_conv_src), err );

        size_t lt_conv_diff_filter_len = dnnLayoutGetMemorySize_F32(lt_conv_diff_filter) / sizeof(float);
        size_t lt_conv_diff_dst_len = dnnLayoutGetMemorySize_F32(lt_conv_diff_dst) / sizeof(float);
        size_t lt_conv_src_len = dnnLayoutGetMemorySize_F32(lt_conv_src) / sizeof(float);

        for (size_t i = 0; i < lt_conv_diff_filter_len; i++) resconv[dnnResourceDiffFilter][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_diff_dst_len; i++) resconv[dnnResourceDiffDst][i] = (float)drand48();
        for (size_t i = 0; i < lt_conv_src_len; i++) resconv[dnnResourceSrc][i] = (float)drand48();
    } else {
        ret = -1;
        goto bail_out;
    }

    CHECK_ERR( dnnExecute_F32(conv, (void**)resconv), err );
    double time_a[1024];
    for (int i = 0; i < NTIMES; i++) {
        double r_start =  getWallClockSeconds();
        CHECK_ERR( dnnExecute_F32(conv, (void**)resconv), err );
        double r_stop =  getWallClockSeconds();
        time_a[i] = (r_stop - r_start);
    }
    double avg_time = time_a[0];
    double min_time = time_a[0];
    for (int i = 1; i < NTIMES; i++) {
        if (min_time > time_a[i]) min_time = time_a[i];
        avg_time += time_a[i];
    }
    avg_time /= (double)NTIMES;

    double min_gflops = ( flops / min_time ) * 1e-9;
    double avg_gflops = ( flops / avg_time ) * 1e-9;
    if      (mode == FWD_CONVOLUTION  ) printf("FWD ");
    else if (mode == BWD_F_CONVOLUTION) printf("BWD_F ");
    else if (mode == BWD_D_CONVOLUTION) printf("BWD_D ");
    if (pad == PAD) printf(" w/ padding in flops :");
    else if (pad == NO_PAD) printf(" w/o padding in flops :");
    printf("min(ms) %.2f; min(gflop/s) %.2f; avg(ms) %.2f; avg(gflop/s) %.2f;\n",
        min_time/1e-3, min_gflops, avg_time/1e-3, avg_gflops); fflush(0);

bail_out:
    dnnLayoutDelete_F32(lt_conv_src);
    dnnLayoutDelete_F32(lt_conv_diff_src);
    dnnLayoutDelete_F32(lt_conv_dst);
    dnnLayoutDelete_F32(lt_conv_diff_dst);
    dnnLayoutDelete_F32(lt_conv_filter);
    dnnLayoutDelete_F32(lt_conv_diff_filter);
    dnnLayoutDelete_F32(lt_conv_bias);
    for (int i = 0; i < dnnResourceNumber; i++)
        dnnReleaseBuffer_F32((void *)resconv[i]);
    dnnDelete_F32(conv);
#ifdef MEASURE_BWD_FILT_CONVERSION
    dnnDelete_F32(fwd_conv);
    dnnDelete_F32(filt_cv);
    dnnLayoutDelete_F32(lt_fwd_conv_filter);
    dnnReleaseBuffer_F32((void *)rescv[dnnResourceFrom]);
#endif
    return ret;
}

int main(int argc, char **argv) {
  int err, pad_mode;

  if (argc == 1)
    pad_mode = PAD;
  else {
    if (argc != 2) {
        printf("Usage: <executable> [<flops w/ padding> = 1 | <flops w/o padding> = 0]\n");
        return 0;
    }
    pad_mode =  (size_t)atoi(argv[1]);
    if (pad_mode != PAD && pad_mode != NO_PAD) {
        printf("Usage: <executable> [<flops w/ padding> = 1 | <flops w/o padding> = 0]\n");
        return 0;
    }
  }

  printf (" FWD Convolution \n");
  for (size_t i = 0; i < sizeof(conv_params)/sizeof(conv_params[0]); ++i) {
    printf ("W=%d, H=%d, C=%d, N=%d, K=%d, R=%d, S=%d | ", conv_params[i].w, conv_params[i].h, conv_params[i].ic, conv_params[i].minibatch, conv_params[i].oc, conv_params[i].kw, conv_params[i].kh);
    err = bench_conv(conv_params[i], FWD_CONVOLUTION, pad_mode);
    if (err != E_SUCCESS) { printf("FWD_CONVOLUTION | PAD FAILED\n"); return err; }
  }
  
  printf (" BWD_D Convolution \n");
  for (size_t i = 0; i < sizeof(conv_params)/sizeof(conv_params[0]); ++i) {
    printf ("W=%d, H=%d, C=%d, N=%d, K=%d, R=%d, S=%d | ", conv_params[i].w, conv_params[i].h, conv_params[i].ic, conv_params[i].minibatch, conv_params[i].oc, conv_params[i].kw, conv_params[i].kh);
    err = bench_conv(conv_params[i], BWD_D_CONVOLUTION, pad_mode);
    if (err != E_SUCCESS) { printf("BWD_D_CONVOLUTION | PAD FAILED\n"); return err; }
  }

  printf (" BWD_F Convolution \n");
  for (size_t i = 0; i < sizeof(conv_params)/sizeof(conv_params[0]); ++i) {
    printf ("W=%d, H=%d, C=%d, N=%d, K=%d, R=%d, S=%d | ", conv_params[i].w, conv_params[i].h, conv_params[i].ic, conv_params[i].minibatch, conv_params[i].oc, conv_params[i].kw, conv_params[i].kh);
    err = bench_conv(conv_params[i], BWD_F_CONVOLUTION, pad_mode);
    if (err != E_SUCCESS) { printf("BWD_F_CONVOLUTION | PAD FAILED\n"); return err; }
  }
  return 0;
}

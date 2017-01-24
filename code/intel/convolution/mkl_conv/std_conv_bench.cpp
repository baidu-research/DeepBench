/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <stdio.h>
#include <float.h>
#include <time.h>

#include <stdexcept>
#include <vector>
#include <string>

#ifndef INPUT_H
#error INPUT_H is not defined
#endif
#include INPUT_H

#define FWD_CONVOLUTION   0
#define BWD_F_CONVOLUTION 1
#define BWD_D_CONVOLUTION 2

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(
        int input_dim, int filter_dim, int padd, int stride)
{
    return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// Calculates number of operations. Note that this assumes that the numbers of
// output and input channels *HAVE NOT* been divided by the number of groups
// as MKL and MKL-DNN do; this is so to match the data as defined in input.h
static double calc_flops(bool skip_padding, const conv_problem& prob)
{
    double flops;
    // Recalculate output dims here to reduce the number of params
    int OW = calc_out_dim(prob.w, prob.fw, prob.padd, prob.stride);
    int OH = calc_out_dim(prob.h, prob.fh, prob.padd, prob.stride);
    if (skip_padding) {
        flops = 0;
        for (int oh = 0; oh < OH; ++oh)
        for (int fh = 0; fh < prob.fh; ++fh) {
            int ih = oh * prob.stride + fh - prob.padd;
            if (!(ih >= 0 && ih < prob.h))
                continue;
            for (int ow = 0; ow < OW; ++ow)
            for (int fw = 0; fw < prob.fw; ++fw) {
                int iw = ow * prob.stride + fw - prob.padd;
                flops += (iw >= 0 && iw < prob.w);
            }
        }
    } else
        flops = 1.0 * prob.fw * prob.fh * OW * OH;
    return 2.0 * flops * prob.ic * prob.oc * prob.minibatch;
}

struct bench_result {
    double min_ms, max_gflops;
    double avg_ms, avg_gflops;
};

// Returns milliseconds since the start of the epoch
static inline double ms_timer()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (1000000000ll * tv.tv_sec + tv.tv_nsec) / 1e6;
}

// Benchmarking loop
template <typename Func>
static inline bench_result timeit(int niters, double flops, Func func)
{
    const double max_ms_total = 3E3; // max milliseconds per problem
    func(); // Warmup
    bench_result result = {DBL_MAX, 0, 0, 0};
    int iters_done = 0;
    for (; iters_done < niters; iters_done++) {
        double ms = ms_timer();
        func();
        ms = ms_timer() - ms;
        result.avg_ms += ms;
        result.min_ms = std::min(result.min_ms, ms);
        if (result.avg_ms > max_ms_total)
            break;
    }
    result.avg_ms /= iters_done + 1;
    result.avg_gflops = flops / result.avg_ms * 1E-6;
    result.max_gflops = flops / result.min_ms * 1E-6;
    return result;
}

static inline void rand_fill(float *data, size_t len)
{
    static bool initialized = false;
    if (!initialized) {
        srand48(1);
        initialized = true;
    }
    for (size_t i = 0; i < len; i++)
        data[i] = drand48();
}

#include "mkl_dnn.h"

#define STR1(x) #x
#define STR(x) STR1(x)

#define CHECK(dnnCall) do { \
    dnnError_t e = dnnCall; \
    if (e != E_SUCCESS) { \
        printf("[%s:%d] %s = %d\n", __FILE__, __LINE__, STR(dnnCall), e); \
        throw std::runtime_error(STR(dnnCall)); \
    } \
} while (0)

static bench_result bench_conv(conv_problem prob, int mode, bool skip_padding)
{
    size_t groups = std::max(1, prob.groups);
    size_t inputSize[] = {prob.w, prob.h, prob.ic, prob.minibatch};
    size_t filterSize[] = {prob.fw, prob.fh,
        prob.ic / groups, prob.oc / groups, groups};
    size_t outputSize[] = {
        calc_out_dim(prob.w, prob.fw, prob.padd, prob.stride),
        calc_out_dim(prob.h, prob.fh, prob.padd, prob.stride),
        prob.oc, prob.minibatch};
    size_t biasSize[] = {prob.oc};
    size_t convolutionStride[] = {prob.stride, prob.stride};
    int inputOffset[] = {-prob.padd, -prob.padd};

    dnnPrimitive_t conv = NULL;
    void* resources[dnnResourceNumber] = {0};

    // Init requested convolution primitive
    std::vector<dnnResourceType_t> active_resource_types;
    if (mode == FWD_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateForwardBias_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceSrc,
            dnnResourceDst, dnnResourceFilter, dnnResourceBias};
    } else if (mode == BWD_D_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateBackwardData_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceDiffSrc,
            dnnResourceDiffDst, dnnResourceFilter};
    } else if (mode == BWD_F_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateBackwardFilter_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceSrc,
            dnnResourceDiffDst, dnnResourceDiffFilter};
    } else
        throw std::runtime_error("Invalid benchmarking mode");

    // Init all resources needed by the current convolution
    for (auto type : active_resource_types) {
        dnnLayout_t layout;
        CHECK(dnnLayoutCreateFromPrimitive_F32(&layout, conv, type));
        CHECK(dnnAllocateBuffer_F32(&resources[type], layout));
        size_t len = dnnLayoutGetMemorySize_F32(layout) / sizeof(float);
        rand_fill(static_cast<float *>(resources[type]), len);
        CHECK(dnnLayoutDelete_F32(layout));
    }

    auto result = timeit(prob.iters, calc_flops(skip_padding, prob),
            [&](){CHECK(dnnExecute_F32(conv, resources));});

    // Release resources
    for (int i = 0; i < dnnResourceNumber; i++)
        dnnReleaseBuffer_F32(resources[i]);
    dnnDelete_F32(conv);

    return result;
}

static void usage()
{
    printf("Usage: <executable> "
            "[<flops w/ padding> = 1 | <flops w/o padding> = 0]\n");
    exit(-1);
}

int main(int argc, char **argv)
{
    if (argc > 3)
        usage();

    bool skip_padding = true;
    if (argc > 1) {
        if (argv[1] == std::string("0"))
            skip_padding = true;
        else if (argv[1] == std::string("1"))
            skip_padding = false;
        else
            usage();
    }

    bool csv_output = false;
    if (argc > 2) {
        if (argv[2] == std::string("--csv-output"))
            csv_output = true;
        else if (argv[2] == std::string("--original-output"))
            csv_output = false;
        else
            usage();
    }

    const char *conv_mode_strs[] = {"FWD", "BWD_F", "BWD_D"};
    const char *skip_padding_strs[]
        = {"w/ padding in flops", "w/o padding in flops"};

    for (auto m : {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION}) {
        if (!csv_output)
            printf(" %s Convolution\n", conv_mode_strs[m]);
        for (const auto& p : conv_problems) {
            auto r = bench_conv(p, m, skip_padding);
            if (csv_output)
                printf("%s,%d,\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e,%e,%e,%e\n",
                        conv_mode_strs[m], skip_padding, p.name,
                        p.minibatch, p.w, p.h, p.ic, p.oc, p.fw, p.fh,
                        p.stride, p.stride, p.padd, p.padd,
                        r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
            else
                printf("W=%d, H=%d, C=%d, N=%d, K=%d, R=%d, S=%d | "
                        "%s %s min(ms) %.2f; max(gflop/s) %.2f;"
                        "avg(ms) %.2f; avg(gflop/s) %.2f;\n",
                        p.w, p.h, p.ic, p.minibatch, p.oc, p.fw, p.fh,
                        conv_mode_strs[m], skip_padding_strs[skip_padding],
                        r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
            fflush(0);
        }
    }

    return 0;
}

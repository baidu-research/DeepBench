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
#include <assert.h>

#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>

struct conv_problem {
    int minibatch;
    int w;
    int h;
    int ic;
    int oc;
    int fw;
    int fh;
    int stride_w, stride_h;
    int pad_w, pad_h;
    int iters;
};

#include "conv_problems.h"

#define FWD_CONVOLUTION   0
#define BWD_F_CONVOLUTION 1
#define BWD_D_CONVOLUTION 2

#define PREC_F32 0
#define PREC_U8S8U8 1
#define PREC_S16S16S32 2

#define TRAINING 0
#define INFERENCE_SERVER 1
#define INFERENCE_DEVICE 2

#define ITERS 1000

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(
        int input_dim, int filter_dim, int padd, int stride)
{
    return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// Calculates number of operations.
static double calc_flops(bool skip_padding, const conv_problem& prob)
{
    double flops;
    // Recalculate output dims here to reduce the number of params
    int OW = calc_out_dim(prob.w, prob.fw, prob.pad_w, prob.stride_w);
    int OH = calc_out_dim(prob.h, prob.fh, prob.pad_h, prob.stride_h);
    if (skip_padding) {
        flops = 0;
        for (int oh = 0; oh < OH; ++oh)
        for (int fh = 0; fh < prob.fh; ++fh) {
            int ih = oh * prob.stride_h + fh - prob.pad_h;
            if (!(ih >= 0 && ih < prob.h))
                continue;
            for (int ow = 0; ow < OW; ++ow)
            for (int fw = 0; fw < prob.fw; ++fw) {
                int iw = ow * prob.stride_w + fw - prob.pad_w;
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
    for (; iters_done < niters && result.avg_ms < max_ms_total; iters_done++) {
        double ms = ms_timer();
        func();
        ms = ms_timer() - ms;
        result.avg_ms += ms;
        result.min_ms = std::min(result.min_ms, ms);
    }
    result.avg_ms /= iters_done;
    result.avg_gflops = flops / result.avg_ms * 1E-6;
    result.max_gflops = flops / result.min_ms * 1E-6;
    return result;
}

template <typename T>
static inline void rand_fill(T *data, size_t size)
{
    static bool initialized = false;
    if (!initialized) {
        srand48(1);
        initialized = true;
    }
#pragma omp parallel for
    for (size_t i = 0; i < size / sizeof(T); i++)
        data[i] = static_cast<T>(drand48());
}

#ifdef USE_MKL

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

static bench_result bench_conv(conv_problem prob,
        int mode, int precision, bool skip_padding)
{
    assert(precision == PREC_F32);

    size_t groups = 1;
    size_t inputSize[] = {prob.w, prob.h, prob.ic, prob.minibatch};
    size_t filterSize[] = {prob.fw, prob.fh,
        prob.ic / groups, prob.oc / groups, groups};
    size_t outputSize[] = {
        calc_out_dim(prob.w, prob.fw, prob.pad_w, prob.stride_w),
        calc_out_dim(prob.h, prob.fh, prob.pad_h, prob.stride_h),
        prob.oc, prob.minibatch};
    size_t biasSize[] = {prob.oc};
    size_t convolutionStride[] = {prob.stride_w, prob.stride_h};
    int inputOffset[] = {-prob.pad_w, -prob.pad_h};

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
        size_t size = dnnLayoutGetMemorySize_F32(layout);
        rand_fill(static_cast<float *>(resources[type]), size);
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
#endif

#ifdef USE_MKLDNN

#define COMPUTE_BWD_BIAS 0

#include "mkldnn.hpp"

using namespace mkldnn;

static bench_result bench_conv(conv_problem prob,
        int mode, int precision, bool skip_padding)
{
    engine eng(engine::kind::cpu, 0);

    int groups = 1;

    memory::data_type src_dt, dst_dt, filter_dt, bias_dt;
    switch (precision) {
    case PREC_U8S8U8:
        src_dt = memory::data_type::u8;
        dst_dt = memory::data_type::u8;
        filter_dt = memory::data_type::s8;
        bias_dt = memory::data_type::s32;
        break;
    case PREC_S16S16S32:
        src_dt = filter_dt = memory::data_type::s16;
        dst_dt = bias_dt = memory::data_type::s32;
        break;
    default:
        src_dt = dst_dt = filter_dt = bias_dt = memory::data_type::f32;
    }

    memory::desc src_d({prob.minibatch, prob.ic, prob.h, prob.w},
            src_dt, memory::format::any);
    memory::desc dst_d({prob.minibatch, prob.oc,
            calc_out_dim(prob.h, prob.fh, prob.pad_h, prob.stride_h),
            calc_out_dim(prob.w, prob.fw, prob.pad_w, prob.stride_w)},
            dst_dt, memory::format::any);
    std::vector<int> fsizes
        = {prob.oc / groups, prob.ic / groups, prob.fh, prob.fw};
    if (groups != 1) fsizes.insert(fsizes.begin(), groups);
    memory::desc filter_d(fsizes, filter_dt, memory::format::any);
    memory::desc bias_d({prob.oc},
            bias_dt, memory::format::any);
    memory::dims strides = {prob.stride_h, prob.stride_w};
    memory::dims padding = {prob.pad_h, prob.pad_w};

    std::shared_ptr<primitive> conv;
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> filter;
    std::shared_ptr<memory> bias;

    auto fwd_conv_pd = convolution_forward::primitive_desc(
            {prop_kind::forward_training, algorithm::convolution_direct,
            src_d, filter_d, bias_d, dst_d,
            strides, padding, padding, padding_kind::zero}, eng);

    if (mode == FWD_CONVOLUTION) {
        src.reset(new memory(fwd_conv_pd.src_primitive_desc()));
        dst.reset(new memory(fwd_conv_pd.dst_primitive_desc()));
        filter.reset(new memory(fwd_conv_pd.weights_primitive_desc()));
        bias.reset(new memory(fwd_conv_pd.bias_primitive_desc()));
        conv.reset(new convolution_forward(fwd_conv_pd,
                    *src, *filter, *bias, *dst));
    } else if (mode == BWD_D_CONVOLUTION) {
        auto bwd_d_conv_pd = convolution_backward_data::primitive_desc(
                {algorithm::convolution_direct, src_d, filter_d, dst_d,
                strides, padding, padding, padding_kind::zero}, eng,
                fwd_conv_pd);
        src.reset(new memory(bwd_d_conv_pd.diff_src_primitive_desc()));
        dst.reset(new memory(bwd_d_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_d_conv_pd.weights_primitive_desc()));
        conv.reset(new convolution_backward_data(bwd_d_conv_pd,
                    *dst, *filter, *src));
    } else if (mode == BWD_F_CONVOLUTION) {
        auto bwd_f_conv_pd = convolution_backward_weights::primitive_desc(
                {algorithm::convolution_direct, src_d, filter_d,
#if COMPUTE_BWD_BIAS
                bias_d,
#endif
                dst_d,
                strides, padding, padding, padding_kind::zero}, eng,
                fwd_conv_pd);
        src.reset(new memory(bwd_f_conv_pd.src_primitive_desc()));
        dst.reset(new memory(bwd_f_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_f_conv_pd.diff_weights_primitive_desc()));
#if COMPUTE_BWD_BIAS
        bias.reset(new memory(bwd_f_conv_pd.diff_bias_primitive_desc()));
        conv.reset(new convolution_backward_weights(bwd_f_conv_pd,
                    *src, *dst, *filter, *bias));
#else
        conv.reset(new convolution_backward_weights(bwd_f_conv_pd,
                    *src, *dst, *filter));
#endif
    } else
        throw std::runtime_error("Invalid benchmarking mode");

    for (const auto &m : {src, dst, filter, bias}) {
        if (!m.get() || !m->get())
            continue;
        void *data = m->get_data_handle();
        auto pd = m->get_primitive_desc();
        size_t size = pd.get_size();
        switch (pd.desc().data.data_type) {
        case memory::data_type::f32:
            rand_fill(static_cast<float *>(data), size);
            break;
        case memory::data_type::u8:
            rand_fill(static_cast<uint8_t *>(data), size);
            break;
        case memory::data_type::s8:
            rand_fill(static_cast<int8_t *>(data), size);
            break;
        case memory::data_type::s16:
            rand_fill(static_cast<int16_t *>(data), size);
            break;
        case memory::data_type::s32:
            rand_fill(static_cast<int32_t *>(data), size);
            break;
        default:
            assert(!"Unsupported data type!\n");
        }
    }

    stream str(stream::kind::eager);
    str.submit({*conv}).wait();

    return timeit(prob.iters, calc_flops(skip_padding, prob),
            [&](){str.rerun().wait();});
}
#endif

static void usage()
{
    printf(
            "Usage: <executable> [OPTIONS]\n"
            "\n"
            "Output control:\n"
            "   --csv-output        Produce CSV output\n"
            "   --original-output   Produce output in the original format\n"
            "\n"
            "Control flops calculations:\n"
            "   --no-skip-padding   Count ops with padding zeroes (default)\n"
            "   --skip-padding      Do not count ops with padding zeroes\n"
            "\n"
            "Precision control:\n"
            "   --f32               32-bit floating point (default)\n"
            "   --u8s8u8            8-bit integers (AVX512VL CPUs)\n"
            "   --s16s16s32         16-bit integers with 32-bit output\n"
            "                       (AVX512_4VNNI CPUs)\n"
            "Problem set control:\n"
            "   --training          Training data set (default)\n"
            "   --inference-server  Server inference data set\n"
            "   --inference-device  Device inference data set\n"
            "\n"
          );
    exit(-1);
}

int main(int argc, char **argv)
{
    bool skip_padding = false;
    bool csv_output = false;
    int precision = PREC_F32;
    std::vector<int> modes
        = {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION};
    int problem_set = TRAINING;

    for(argc--, argv++; argc; argv++, argc--) {
        if (*argv == std::string("--csv-output"))
            csv_output = true;
        else if (*argv == std::string("--original-output"))
            csv_output = false;
        else if (*argv == std::string("--skip-padding"))
            skip_padding = true;
        else if (*argv == std::string("--no-skip-padding"))
            skip_padding = false;
        else if (*argv == std::string("--f32"))
            precision = PREC_F32;
        else if (*argv == std::string("--u8s8u8"))
            precision = PREC_U8S8U8;
        else if (*argv == std::string("--s16s16s32"))
            precision = PREC_S16S16S32;
        else if (*argv == std::string("--inference-device"))
            problem_set = INFERENCE_DEVICE;
        else if (*argv == std::string("--inference-server"))
            problem_set = INFERENCE_SERVER;
        else if (*argv == std::string("--training"))
            problem_set = TRAINING;
        else
            usage();
    }

#ifdef USE_MKL
    if (precision != PREC_F32) {
        printf("MKL version of DeepBench only support F32 precision. "
                "Please use MKL-DNN version instead.\n");
        usage();
    }
#endif

#ifdef USE_MKLDNN
    if (precision != PREC_F32 || problem_set != TRAINING)
        modes = {FWD_CONVOLUTION};
#endif

    const char *conv_mode_strs[] = {"FWD", "BWD_F", "BWD_D"};
    const char *skip_padding_strs[]
        = {"w/ padding in flops", "w/o padding in flops"};

    const auto &problems = (problem_set == TRAINING
            ? training_set
            : (problem_set == INFERENCE_DEVICE
                ? inference_device_set
                : inference_server_set));

    for (auto m : modes) {
        if (!csv_output)
            printf(" %s Convolution\n", conv_mode_strs[m]);
        for (const auto& problem : problems) {
            conv_problem p;
            std::tie(p.w, p.h, p.ic, p.minibatch, p.oc, p.fw, p.fh,
                    p.pad_w, p.pad_h, p.stride_w, p.stride_h) = problem;
            p.iters = ITERS;
            auto r = bench_conv(p, m, precision, skip_padding);
            if (csv_output)
                printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e,%e,%e,%e\n",
                        conv_mode_strs[m], skip_padding,
                        p.minibatch, p.w, p.h, p.ic, p.oc, p.fw, p.fh,
                        p.stride_w, p.stride_h, p.pad_w, p.pad_h,
                        r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
            else
                printf("W=%d, H=%d, C=%d, N=%d, K=%d, S=%d, R=%d | "
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

#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"
#include "cudnn_helper.h"

class cudnnCNN {
    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> h_desc_;

    FilterDescriptor4d<float> w_desc_;

    std::vector<int> output_dims_;
    int num_repeats_;

    size_t fwd_workspace_size_;
    size_t bwd_inputs_workspace_size_;
    size_t bwd_params_workspace_size_;

    Tensor<float> fwd_workspace_;
    Tensor<float> bwd_inputs_workspace_;
    Tensor<float> bwd_params_workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_inputs_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_params_algo_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    ConvolutionDescriptor conv_desc_;
    CudnnHandle cudnn_handle_;

public:

    cudnnCNN(int w, int h, int c, int n, int k, int r, int s,
             int pad_w, int pad_h, int wstride, int hstride)
             :
        cudnn_handle_(),
        x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
        w_desc_(CUDNN_TENSOR_NCHW, k, c, r, s),
        conv_desc_(pad_h, pad_w, hstride, wstride)
    {
        int out_h, out_w, out_c, out_n;

        // Get output dimensions
        CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_desc_.desc(),
                                                                x_desc_.desc(),
                                                                w_desc_.desc(),
                                                                &out_n,
                                                                &out_c,
                                                                &out_h,
                                                                &out_w));

        h_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW, out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};

        // Pick forward convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                              x_desc_.desc(),
                                                              w_desc_.desc(),
                                                              conv_desc_.desc(),
                                                              h_desc_.desc(),
                                                              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                              0,
                                                              &fwd_algo_));

        // Set fwd workspace size
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_.handle(),
                                                                  x_desc_.desc(),
                                                                  w_desc_.desc(),
                                                                  conv_desc_.desc(),
                                                                  h_desc_.desc(),
                                                                  fwd_algo_,
                                                                  &fwd_workspace_size_));

        fwd_workspace_ = zeros(std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});

        // Pick backward convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_.handle(),
                                                                     x_desc_.desc(),
                                                                     h_desc_.desc(),
                                                                     conv_desc_.desc(),
                                                                     w_desc_.desc(),
                                                                     CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                                     0,
                                                                     &bwd_params_algo_));

        // Backward params workspace
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_.handle(),
                                                                         x_desc_.desc(),
                                                                         h_desc_.desc(),
                                                                         conv_desc_.desc(),
                                                                         w_desc_.desc(),
                                                                         bwd_params_algo_,
                                                                         &bwd_params_workspace_size_));



        bwd_params_workspace_ = zeros(std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1});

        // Pick backward wrt inputs convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle_.handle(),
                                                                   w_desc_.desc(),
                                                                   h_desc_.desc(),
                                                                   conv_desc_.desc(),
                                                                   x_desc_.desc(),
                                                                   CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                                   0,
                                                                   &bwd_inputs_algo_));

        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_.handle(),
                                                                       w_desc_.desc(),
                                                                       h_desc_.desc(),
                                                                       conv_desc_.desc(),
                                                                       x_desc_.desc(),
                                                                       bwd_inputs_algo_,
                                                                       &bwd_inputs_workspace_size_));

        bwd_inputs_workspace_ = zeros(std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1});

    }

    std::vector<int> get_output_dims() { return output_dims_; }

    std::string get_fwd_algo_string() {
        if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
            return "IMPLICIT_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
            return "IMPLICIT_PRECOMP_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_GEMM) 
            return "GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
            return "DIRECT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            return "FFT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
            return "FFT_TILING";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
            return "WINOGRAD";
        else {
            std::stringstream ss;
            ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
            throw std::runtime_error(ss.str());
        }
    }


    void forward(Tensor<float> x, Tensor<float> filter, Tensor<float> h) {

        // Convolution forward.
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn_handle_.handle(),
                                                  &alpha_,
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  w_desc_.desc(),
                                                  filter.begin(),
                                                  conv_desc_.desc(),
                                                  fwd_algo_,
                                                  fwd_workspace_.begin(),
                                                  fwd_workspace_size_,
                                                  &beta_,
                                                  h_desc_.desc(),
                                                  h.begin()));

    }

    void backward_params(Tensor<float> x, Tensor<float> delta, Tensor<float> dW) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(cudnn_handle_.handle(),
                                                         &alpha_,
                                                         x_desc_.desc(),
                                                         x.begin(),
                                                         h_desc_.desc(),
                                                         delta.begin(),
                                                         conv_desc_.desc(),
                                                         bwd_params_algo_,
                                                         bwd_params_workspace_.begin(),
                                                         bwd_params_workspace_size_,
                                                         &beta_,
                                                         w_desc_.desc(),
                                                         dW.begin()));


    }

    void backward_inputs(Tensor<float> filter, Tensor<float> delta, Tensor<float> dX) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(cudnn_handle_.handle(),
                                                      &alpha_,
                                                      w_desc_.desc(),
                                                      filter.begin(),
                                                      h_desc_.desc(),
                                                      delta.begin(),
                                                      conv_desc_.desc(),
                                                      bwd_inputs_algo_,
                                                      bwd_inputs_workspace_.begin(),
                                                      bwd_inputs_workspace_size_,
                                                      &beta_,
                                                      x_desc_.desc(),
                                                      dX.begin()));

    }
};

std::tuple<int, int, int, std::string> time_cnn(
         int k, int c, int r, int s,
         int n, int h, int w,
         int pad_h, int pad_w,
         int hstride, int wstride,
         int num_repeats,
         curandGenerator_t curand_gen
        ) {

    cudnnCNN cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride);

    // Allocate memory for filter
    auto filter = rand(std::vector<int>{r, s, c, k}, curand_gen);
    
    // Allocate memory for input
    auto input = rand(std::vector<int>{w, h, c, n}, curand_gen);

    // Allocate memory for output tensor
    auto output = zeros(cnn.get_output_dims());


    std::string fwd_algo_s = cnn.get_fwd_algo_string();

    //Warm up
    cnn.forward(input, filter, output);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        cnn.forward(input, filter, output);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    // Allocate memory for backward pass wrt weights
    auto delta = rand(cnn.get_output_dims(), curand_gen);
    auto dW = zeros(std::vector<int>{r, s, c, k});

    // Warm up backward
    cnn.backward_params(input, delta, dW);

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_params(input, delta, dW);
    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    int bwd_params_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    //Allocate memory for backward pass wrt inputs
    auto dX = zeros(std::vector<int>{w, h, c, n});

    //Warm up backward inputs
    cnn.backward_inputs(filter, delta, dX);

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_inputs(filter, delta, dX);

    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    int bwd_inputs_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    return std::tuple<int, int, int, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);

}

int main(int argc, char **argv) {

    int num_repeats = 100;

    // Handles to various cuda libraries, structures
    curandGenerator_t curand_gen;


    cudaFree(0);

    if (argc > 1)
        num_repeats = atoi(argv[1]);


    // Initialize curand_gen and set appropriate seed.
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);


    // Vector saves w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems = {
        std::make_tuple(700, 161, 1, 4, 32, 5, 20, 0, 0, 2, 2),
        std::make_tuple(700, 161, 1, 8, 32, 5, 20, 0, 0, 2, 2),
        std::make_tuple(700, 161, 1, 16, 32, 5, 20, 0, 0, 2, 2),
        std::make_tuple(700, 161, 1, 32, 32, 5, 20, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 4, 32, 5, 10, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 8, 32, 5, 10, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 16, 32, 5, 10, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 32, 32, 5, 10, 0, 0, 2, 2),
        std::make_tuple(480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1),
        std::make_tuple(240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1),
        std::make_tuple(120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2),
        std::make_tuple(54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 256, 8, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 8, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(112, 112, 64, 8, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 128, 8, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 256, 8, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 512, 8, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 8, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 16, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(112, 112, 64, 16, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 128, 16, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 256, 16, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 512, 16, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 16, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 16, 64, 7, 7, 3, 3, 2, 2),
        std::make_tuple(28, 28, 192, 16, 32, 5, 5, 2, 2, 1, 1),
        std::make_tuple(28, 28, 192, 16, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 512, 16, 48, 5, 5, 2, 2, 1, 1),
        std::make_tuple(14, 14, 512, 16, 192, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 16, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 16, 128, 5, 5, 2, 2, 1, 1)
    };

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "   w      h      c      n      k      r      s    pad_w  pad_h    stride_w  stride_h    fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)   fwd_algo " << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');

    for (const auto &problem : problems) {

        // Filter parameters
        int k, c, r, s;

        // Input parameters
        int n, w, h;

        // Padding
        int pad_w, pad_h;

        // Stride
        int wstride, hstride;

        std::tie(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride) = problem;

        int fwd_time, bwd_inputs_time, bwd_params_time;
        std::string fwd_algo_s;

        std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) = 
            time_cnn(k, c, r, s, n, h, w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen);

        std::cout << std::setw(5) << w;
        std::cout << std::setw(7) << h;
        std::cout << std::setw(7) << c;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << r;
        std::cout << std::setw(7) << s;
        std::cout << std::setw(7) << pad_w;
        std::cout << std::setw(8) << pad_h;
        std::cout << std::setw(10) << wstride;
        std::cout << std::setw(10) << hstride;
        std::cout << std::setw(14) << std::setprecision(7) << fwd_time;
        std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
        std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
        std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;

        std::cout << std::setw(25) << fwd_algo_s;

        std::cout << std::endl;

    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
    return 0;

}

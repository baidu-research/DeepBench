#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "tensor.h"
#include "miopen_helper.h"

class miopenCNN {
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

    Tensor<float> h;

    miopenConvFwdAlgorithm_t fwd_algo_;
    miopenConvBwdDataAlgorithm_t bwd_inputs_algo_;
    miopenConvBwdWeightsAlgorithm_t bwd_params_algo_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    ConvolutionDescriptor conv_desc_;
    MIOpenHandle miopen_handle_;

public:

    miopenCNN(int _w, int _h, int c, int n, int k, int r, int s,
             int pad_w, int pad_h, int wstride, int hstride, Tensor<float> x, Tensor<float> w)
             :
        miopen_handle_(),
        x_desc_(n, c, _h, _w),
        w_desc_(k, c, r, s),
        conv_desc_(pad_h, pad_w, hstride, wstride)
    {
        int out_h, out_w, out_c, out_n;

        // Get output dimensions
        CHECK_MIOPEN_ERROR(miopenGetConvolutionForwardOutputDim(conv_desc_.desc(),
                                                                x_desc_.desc(),
                                                                w_desc_.desc(),
                                                                &out_n,
                                                                &out_c,
                                                                &out_h,
                                                                &out_w));

        h_desc_ = TensorDescriptor4d<float>(out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};

        h = zeros(output_dims_);


        // Set fwd workspace size
        CHECK_MIOPEN_ERROR(miopenConvolutionForwardGetWorkSpaceSize(
                    miopen_handle_.handle(),
                    w_desc_.desc(),
                                                          x_desc_.desc(),
                                                          conv_desc_.desc(),
                                                          h_desc_.desc(),
                                                          &fwd_workspace_size_));

        std::vector<int> u = std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1};

        fwd_workspace_ = zeros(u);

        const int requestAlgoCount = 1;
        int returnedAlgoCount;
        miopenConvAlgoPerf_t perfResults;

        CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(
          miopen_handle_.handle(),
          x_desc_.desc(),
          x.begin(),
          w_desc_.desc(),
          w.begin(),
          conv_desc_.desc(),
          h_desc_.desc(),
          h.begin(),
          requestAlgoCount,
          &returnedAlgoCount,
          &perfResults,
          fwd_workspace_.begin(),
          fwd_workspace_size_,
          false
        ));

        fwd_algo_ = perfResults.fwd_algo;


    CHECK_MIOPEN_ERROR(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                miopen_handle_.handle(),
                                                  h_desc_.desc(),
                                                  x_desc_.desc(),
                                                  conv_desc_.desc(),
                                                  w_desc_.desc(),
                                                  &bwd_params_workspace_size_));

    u = std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1};
    bwd_params_workspace_ = zeros(u);

    CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardWeightsAlgorithm(
      miopen_handle_.handle(),
      h_desc_.desc(),
      h.begin(),
      x_desc_.desc(),
      x.begin(),
      conv_desc_.desc(),
      w_desc_.desc(),
      w.begin(),
      requestAlgoCount,
      &returnedAlgoCount,
      &perfResults,
      bwd_params_workspace_.begin(),
      bwd_params_workspace_size_,
      false
    ));

    bwd_params_algo_ = perfResults.bwd_weights_algo;

        CHECK_MIOPEN_ERROR(miopenFindConvolutionBackwardDataAlgorithm(
          miopen_handle_.handle(),
          h_desc_.desc(),
          h.begin(),
          w_desc_.desc(),
          w.begin(),
          conv_desc_.desc(),
          x_desc_.desc(),
          x.begin(),
          requestAlgoCount,
          &returnedAlgoCount,
          &perfResults,
          nullptr,
          0,
          false
        ));

        bwd_inputs_workspace_size_ = 0;
        bwd_inputs_algo_ = perfResults.bwd_data_algo;

    }

    Tensor<float> getOutputTensor(){ return h; }

    std::vector<int> get_output_dims() { return output_dims_; }

    std::string get_fwd_algo_string() {
        if (fwd_algo_ == miopenConvolutionFwdAlgoGEMM)
            return " ConvolutionFwdAlgoGEMM";
        else if (fwd_algo_ == miopenConvolutionFwdAlgoDirect)
            return " ConvolutionFwdAlgoDirect";
        else if (fwd_algo_ == miopenConvolutionFwdAlgoFFT)
            return " ConvolutionFwdAlgoFFT";
        else if (fwd_algo_ == miopenConvolutionFwdAlgoWinograd)
            return " ConvolutionFwdAlgoWinograd";
        else {
            std::stringstream ss;
            ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
            throw std::runtime_error(ss.str());
        }
    }


    void forward(Tensor<float> x, Tensor<float> filter, Tensor<float> h) {

        // Convolution forward.
        CHECK_MIOPEN_ERROR(miopenConvolutionForward(miopen_handle_.handle(),
                                                  &alpha_,
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  w_desc_.desc(),
                                                  filter.begin(),
                                                  conv_desc_.desc(),
                                                  fwd_algo_,
                                                  &beta_,
                                                  h_desc_.desc(),
                                                  h.begin(),
                                                  fwd_workspace_.begin(),
                                                  fwd_workspace_size_
                                                ));

    }

    void backward_params(Tensor<float> x, Tensor<float> delta, Tensor<float> dW) {

        CHECK_MIOPEN_ERROR(miopenConvolutionBackwardWeights(miopen_handle_.handle(),
                                                         &alpha_,
                                                         h_desc_.desc(),
                                                         delta.begin(),
                                                         x_desc_.desc(),
                                                         x.begin(),
                                                         conv_desc_.desc(),
                                                         bwd_params_algo_,
                                                         &beta_,
                                                         w_desc_.desc(),
                                                         dW.begin(),
                                                         bwd_params_workspace_.begin(),
                                                         bwd_params_workspace_size_
                                                       ));


    }

    void backward_inputs(Tensor<float> filter, Tensor<float> delta, Tensor<float> dX) {

        CHECK_MIOPEN_ERROR(miopenConvolutionBackwardData(miopen_handle_.handle(),
                                                      &alpha_,
                                                      h_desc_.desc(),
                                                      delta.begin(),
                                                      w_desc_.desc(),
                                                      filter.begin(),
                                                      conv_desc_.desc(),
                                                      bwd_inputs_algo_,
                                                      &beta_,
                                                      x_desc_.desc(),
                                                      dX.begin(),
                                                      nullptr,
                                                      0
                                                    ));

    }
};

std::tuple<int, int, int, std::string> time_cnn(
         int k, int c, int r, int s,
         int n, int h, int w,
         int pad_h, int pad_w,
         int hstride, int wstride,
         int num_repeats
        ) {


    // Allocate memory for filter
    auto filter = rand(std::vector<int>{r, s, c, k});

    // Allocate memory for input
    auto input = rand(std::vector<int>{w, h, c, n});
    miopenCNN cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, input, filter);

    // Allocate memory for output tensor
    auto output = cnn.getOutputTensor();

    std::string fwd_algo_s = cnn.get_fwd_algo_string();

    //Warm up
    cnn.forward(input, filter, output);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        cnn.forward(input, filter, output);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    // Allocate memory for backward pass wrt weights
    auto delta = rand(cnn.get_output_dims());
    auto dW = zeros(std::vector<int>{r, s, c, k});

    // Warm up backward
    cnn.backward_params(input, delta, dW);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_params(input, delta, dW);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    int bwd_params_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    //Allocate memory for backward pass wrt inputs
    auto dX = zeros(std::vector<int>{w, h, c, n});

    //Warm up backward inputs
    cnn.backward_inputs(filter, delta, dX);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        // Backward pass wrt weights
        cnn.backward_inputs(filter, delta, dX);

    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    int bwd_inputs_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    return std::tuple<int, int, int, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);

}

int main(int argc, char **argv) {

    int num_repeats = 100;

    hipFree(0);

    if (argc > 1)
        num_repeats = atoi(argv[1]);

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

    int total_fwd_time=0, total_bwd_inputs_time=0, total_bwd_params_time=0;
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
            time_cnn(k, c, r, s, n, h, w, pad_h, pad_w, hstride, wstride, num_repeats);

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

    	total_fwd_time += fwd_time;
    	total_bwd_inputs_time += bwd_inputs_time;
    	total_bwd_params_time += bwd_params_time;

    }

    std::cout << std::setw(82) << "Totals" ;
    std::cout << std::setw(14) << std::setprecision(7) << total_fwd_time;
    std::cout << std::setw(24) << std::setprecision(7) << total_bwd_inputs_time;
    std::cout << std::setw(24) << std::setprecision(7) << total_bwd_params_time;
    std::cout << std::setw(19) << std::setprecision(8) << total_fwd_time + total_bwd_inputs_time + total_bwd_params_time;
    std::cout << std::endl;

    return 0;

}

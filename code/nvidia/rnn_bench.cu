#include <chrono>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"
#include "cudnn_helper.h"

cudnnHandle_t cudnn_handle;
curandGenerator_t curand_gen;


class cudnnDropout {
    std::shared_ptr<cudnnDropoutDescriptor_t> dropout_desc_;
    std::shared_ptr<Tensor<uint8_t>> dropout_state_;

    struct DropoutDeleter {
        void operator()(cudnnDropoutDescriptor_t * dropout_desc) {
            cudnnDestroyDropoutDescriptor(*dropout_desc);
            delete dropout_desc;
        }
    };

    public:

    cudnnDropout(float dropout_percentage) : dropout_desc_(new cudnnDropoutDescriptor_t,
                                                           DropoutDeleter()) {
        size_t dropoutStateSize;
        CHECK_CUDNN_ERROR(cudnnCreateDropoutDescriptor(dropout_desc_.get()));
        CHECK_CUDNN_ERROR(cudnnDropoutGetStatesSize(cudnn_handle, &dropoutStateSize));

        dropout_state_.reset(new Tensor<uint8_t>(std::vector<int>{static_cast<int>(dropoutStateSize), 1}));

        CHECK_CUDNN_ERROR(cudnnSetDropoutDescriptor(*dropout_desc_,
                                                    cudnn_handle,
                                                    dropout_percentage,
                                                    dropout_state_->begin(),
                                                    dropoutStateSize,
                                                    0ULL) );
    }

    cudnnDropoutDescriptor_t desc() const { return *dropout_desc_; }
};

class cudnnRNN {
    RNNDescriptor<float> rnn_desc_;
    FilterDescriptorNd<float> wDesc_;
    cudnnDropout dropout_;

    int time_steps_;

    TensorDescriptorNdArray<float> xDescArray_;
    TensorDescriptorNdArray<float> yDescArray_;
    TensorDescriptorNdArray<float> dxDescArray_;
    TensorDescriptorNdArray<float> dyDescArray_;

    TensorDescriptorNd<float> hx_desc_;
    TensorDescriptorNd<float> hy_desc_;
    TensorDescriptorNd<float> dhx_desc_;
    TensorDescriptorNd<float> dhy_desc_;
    TensorDescriptorNd<float> cx_desc_;
    TensorDescriptorNd<float> cy_desc_;
    TensorDescriptorNd<float> dcx_desc_;
    TensorDescriptorNd<float> dcy_desc_;

    size_t weight_size_;
    size_t workspace_size_;
    size_t train_size_;

    Tensor<float> weights_;
    Tensor<float> workspace_;
    Tensor<float> trainspace_;

    public:

    cudnnRNN(int hidden_size, int batch_size, int time_steps, const std::string& rnn_type) :
        dropout_(0.f), time_steps_(time_steps),
        xDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        yDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dxDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dyDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        hx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        hy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1})
        {


            rnn_desc_ = RNNDescriptor<float>(hidden_size,
                                             1,
                                             dropout_.desc(),
                                             CUDNN_SKIP_INPUT,
                                             CUDNN_UNIDIRECTIONAL,
                                             rnn_type);

            CHECK_CUDNN_ERROR( cudnnGetRNNParamsSize(cudnn_handle,
                                                     rnn_desc_.desc(),
                                                     xDescArray_.ptr()[0],
                                                     &weight_size_,
                                                     CUDNN_DATA_FLOAT) );

            weights_ = rand(std::vector<int>{static_cast<int>(weight_size_ / sizeof(float)), 1}, curand_gen);


            std::vector<int> dim = {weights_.size(), 1, 1};
            wDesc_ = FilterDescriptorNd<float>(CUDNN_TENSOR_NCHW, dim);

            CHECK_CUDNN_ERROR( cudnnGetRNNWorkspaceSize(cudnn_handle,
                                                        rnn_desc_.desc(),
                                                        time_steps,
                                                        xDescArray_.ptr(),
                                                        &workspace_size_) );

            workspace_ = zeros(std::vector<int>{static_cast<int>(workspace_size_ / sizeof(float)), 1});

            CHECK_CUDNN_ERROR( cudnnGetRNNTrainingReserveSize(cudnn_handle,
                                                              rnn_desc_.desc(),
                                                              time_steps,
                                                              xDescArray_.ptr(),
                                                              &train_size_) );
            trainspace_ = zeros(std::vector<int>{static_cast<int>(train_size_ / sizeof(float)), 1});
        }
        void forward(Tensor<float> x, Tensor<float> hx, Tensor<float> cx,
                     Tensor<float> y, Tensor<float> hy, Tensor<float> cy) {
            CHECK_CUDNN_ERROR( cudnnRNNForwardTraining(cudnn_handle,
                                                       rnn_desc_.desc(),
                                                       time_steps_,
                                                       xDescArray_.ptr(),
                                                       (void *)x.begin(),
                                                       hx_desc_.desc(),
                                                       (void *)hx.begin(),
                                                       cx_desc_.desc(),
                                                       (void *)cx.begin(),
                                                       wDesc_.desc(),
                                                       (void *)weights_.begin(),
                                                       yDescArray_.ptr(),
                                                       (void *)y.begin(),
                                                       hy_desc_.desc(),
                                                       (void *)hy.begin(),
                                                       cy_desc_.desc(),
                                                       (void *)cy.begin(),
                                                       (void *)workspace_.begin(),
                                                       workspace_size_,
                                                       (void *)trainspace_.begin(),
                                                       train_size_) );
        }
        void backward_data(Tensor<float> y, Tensor<float> dy, Tensor<float> dhy,
                           Tensor<float> dcy, Tensor<float> hx, Tensor<float> cx,
                           Tensor<float> dx, Tensor<float> dhx, Tensor<float> dcx) {
            CHECK_CUDNN_ERROR( cudnnRNNBackwardData(cudnn_handle,
                                                    rnn_desc_.desc(),
                                                    time_steps_,
                                                    yDescArray_.ptr(),
                                                    (void *)y.begin(),
                                                    dyDescArray_.ptr(),
                                                    (void *)dy.begin(),
                                                    dhy_desc_.desc(),
                                                    (void *)dhy.begin(),
                                                    dcy_desc_.desc(),
                                                    (void *)dcy.begin(),
                                                    wDesc_.desc(),
                                                    (void *)weights_.begin(),
                                                    hx_desc_.desc(),
                                                    (void *)hx.begin(),
                                                    cx_desc_.desc(),
                                                    (void *)cx.begin(),
                                                    dxDescArray_.ptr(),
                                                    (void *)dx.begin(),
                                                    dhx_desc_.desc(),
                                                    (void *)dhx.begin(),
                                                    dcx_desc_.desc(),
                                                    (void *)dcx.begin(),
                                                    (void *)workspace_.begin(),
                                                    workspace_size_,
                                                    (void *)trainspace_.begin(),
                                                    train_size_) );
        }
};

std::tuple<int, int> time_rnn(int hidden_size,
                              int batch_size,
                              int time_steps,
                              const std::string& type) {

    cudnnRNN rnn(hidden_size, batch_size, time_steps, type);

    auto x  = rand({hidden_size, batch_size * time_steps}, curand_gen);
    auto y  = rand({hidden_size, batch_size * time_steps}, curand_gen);
    auto dx = rand({hidden_size, batch_size * time_steps}, curand_gen);
    auto dy = rand({hidden_size, batch_size * time_steps}, curand_gen);

    auto hx = rand({hidden_size, batch_size}, curand_gen);
    auto hy = rand({hidden_size, batch_size}, curand_gen);
    auto cx = rand({hidden_size, batch_size}, curand_gen);
    auto cy = rand({hidden_size, batch_size}, curand_gen);
    auto dhx = rand({hidden_size, batch_size}, curand_gen);
    auto dhy = rand({hidden_size, batch_size}, curand_gen);
    auto dcx = rand({hidden_size, batch_size}, curand_gen);
    auto dcy = rand({hidden_size, batch_size}, curand_gen);

    int numRepeats = 100;

    //Warm up
    rnn.forward(x, hx, cx, y, hy, cy);

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        rnn.forward(x, hx, cx, y, hy, cy);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    auto forward_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;

    //Warm up
    rnn.backward_data(y, dy, dhy, dcy,
                      hx, cx, dx, dhx, dcx);

    cudaDeviceSynchronize();

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        rnn.backward_data(y, dy, dhy, dcy,
                          hx, cx, dx, dhx, dcx);
    }
    cudaDeviceSynchronize();

    end = std::chrono::steady_clock::now();
    auto backward_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;

    return std::make_tuple(static_cast<int>(forward_time),
                           static_cast<int>(backward_time));

}

int main(int argc, char **argv) {
    cudaFree(0);
    CHECK_CUDNN_ERROR( cudnnCreate(&cudnn_handle) );

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::vector<std::tuple<int, int, int, bool>> problems  = {
        std::make_tuple(1760, 16, 50, false),
        std::make_tuple(1760, 32, 50, false),
        std::make_tuple(1760, 64, 50, false),
        std::make_tuple(1760, 128, 50, false),
        std::make_tuple(2048, 16, 50, false),
        std::make_tuple(2048, 32, 50, false),
        std::make_tuple(2048, 64, 50, false),
        std::make_tuple(2048, 128, 50, false),
        std::make_tuple(2560, 16, 50, false),
        std::make_tuple(2560, 32, 50, false),
        std::make_tuple(2560, 64, 50, false),
        std::make_tuple(2560, 128, 50, false),
        std::make_tuple(512, 16, 25, true),
        std::make_tuple(512, 32, 25, true),
        std::make_tuple(512, 64, 25, true),
        std::make_tuple(512, 128, 25, true),
        std::make_tuple(1024, 16, 25, true),
        std::make_tuple(1024, 32, 25, true),
        std::make_tuple(1024, 64, 25, true),
        std::make_tuple(1024, 128, 25, true),
        std::make_tuple(2048, 16, 25, true),
        std::make_tuple(2048, 32, 25, true),
        std::make_tuple(2048, 64, 25, true),
        std::make_tuple(2048, 128, 25, true),
        std::make_tuple(4096, 16, 25, true),
        std::make_tuple(4096, 32, 25, true),
        std::make_tuple(4096, 64, 25, true),
        std::make_tuple(4096, 128, 25, true)
    };

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    type    hidden   N     timesteps      fwd_time (usec)   bwd_time (usec)" << std::endl;
    for (const auto &problem : problems) {
        int hidden_state, batch_size, time_steps;
        bool lstm;
        std::tie(hidden_state, batch_size, time_steps, lstm) = problem;
        std::string type = lstm ? "lstm" : "vanilla";

        std::cout << std::setw(8) << type;
        std::cout << std::setw(8) << hidden_state;
        std::cout << std::setw(8) << batch_size;
        std::cout << std::setw(8) << 25;
        int fwd_time, bwd_time;
        std::tie(fwd_time, bwd_time) = time_rnn(hidden_state,
                                                batch_size,
                                                time_steps,
                                                type);
        std::cout << std::setw(18) << fwd_time;
        std::cout << std::setw(18) << bwd_time;
        std::cout << std::endl;
    }

    cudnnDestroy(cudnn_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}

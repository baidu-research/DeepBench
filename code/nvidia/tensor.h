#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);

        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

Tensor<float> fill(std::vector<int> dims, float val) {
     Tensor<float> tensor(dims);
     thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                  thrust::device_ptr<float>(tensor.end()), val);
     return tensor;
}

Tensor<float> zeros(std::vector<int> dims) {
    Tensor<float> tensor(dims);
    thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                 thrust::device_ptr<float>(tensor.end()), 0.f);
    return tensor;
}

Tensor<float> rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<float> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

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

template <typename T>
Tensor<T> fill(std::vector<int> dims, float val) {
     Tensor<T> tensor(dims);
     thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                  thrust::device_ptr<T>(tensor.end()), val);
     return tensor;
}

template <typename T>
Tensor<T> zeros(std::vector<int> dims) {
    Tensor<T> tensor(dims);
    thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                 thrust::device_ptr<T>(tensor.end()), 0.f);
    return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<T> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

template <typename T>
typename std::enable_if<!(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen) {

    Tensor<T> tensor(dims);
    Tensor<float> tensor_f(dims);
    curandGenerateUniform(curand_gen, tensor_f.begin(), tensor_f.size());

    thrust::copy(thrust::device_ptr<float>(tensor_f.begin()),
                 thrust::device_ptr<float>(tensor_f.end()),
                 thrust::device_ptr<T>(tensor.begin()));

    return tensor;
}

void pad_dim(int & dim, int pad_v) {
    assert(pad_v > 0);
    if (dim % pad_v) {
        int pad = pad_v - dim%pad_v;
        dim += pad;
    }
}

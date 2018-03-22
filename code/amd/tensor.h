#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <cstdlib>

#include <hip/hip_runtime_api.h>

template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteDevPtr {
        void operator()(T *p) const {
            hipFree(p);
        }
    };


public:
    std::shared_ptr<T> ptr_;

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        hipMalloc(&tmp_ptr, sizeof(T) * size_);

        ptr_.reset(tmp_ptr, deleteDevPtr());
    }

    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

Tensor<float> fill(std::vector<int> dims, float val) {
     Tensor<float> tensor(dims);
     size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
     std::vector<float> host_ptr(d);
     std::fill(host_ptr.begin(), host_ptr.end(), val);
     hipMemcpy(tensor.ptr_.get(), host_ptr.data(), d*sizeof(float), hipMemcpyHostToDevice);
     return tensor;
}

Tensor<float> zeros(std::vector<int> dims) {

    Tensor<float> tensor(dims);
    size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    hipMemset(tensor.ptr_.get(), 0, d*sizeof(float));
    return tensor;
}

Tensor<float> rand(std::vector<int> dims) {
    Tensor<float> tensor(dims);
    size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    std::vector<float> host_ptr(d);
    std::srand(std::time(0));
    for(int i=0;i<d;i++) {
      host_ptr[i] = std::rand();
    }
    hipMemcpy(tensor.ptr_.get(), host_ptr.data(), d*sizeof(float), hipMemcpyHostToDevice);
    return tensor;
}

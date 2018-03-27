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

template<typename T>
Tensor<T> fill(std::vector<int> dims, T val) {
     Tensor<T> tensor(dims);
     size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
     std::vector<T> host_ptr(d);
     std::fill(host_ptr.begin(), host_ptr.end(), val);
     hipMemcpy(tensor.ptr_.get(), host_ptr.data(), d*sizeof(T), hipMemcpyHostToDevice);
     return tensor;
}

template<typename T>
Tensor<T> zeros(std::vector<int> dims)
{
    Tensor<T> tensor(dims);
    size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    hipMemset(tensor.ptr_.get(), 0, d*sizeof(T));
    return tensor;
}

template<typename T>
Tensor<T> rand(std::vector<int> dims) 
{
    Tensor<T> tensor(dims);
    size_t d = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    std::vector<T> host_ptr(d);
    std::srand(std::time(0));
    for(int i=0;i<d;i++) 
    {
      host_ptr[i] = std::rand();
    }
    hipMemcpy(tensor.ptr_.get(), host_ptr.data(), d*sizeof(T), hipMemcpyHostToDevice);
    return tensor;
}

#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include <miopen/miopen.h>

#include "hip_helper.h"

void throw_miopen_err(miopenStatus_t status, int line, const char* filename) {
    if (status != miopenStatusSuccess) {
        std::stringstream ss;
        ss << "MIOPEN failure: " << status <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_MIOPEN_ERROR(status) throw_miopen_err(status, __LINE__, __FILE__)

class MIOpenHandle {
    hipStream_t 	            stream_;
    std::shared_ptr<miopenHandle_t> handle_;

    struct MIOpenHandleDeleter {
        void operator()(miopenHandle_t * handle) {
            miopenDestroy(*handle);
            delete handle;
        }
    };

public:
    MIOpenHandle() : handle_(new miopenHandle_t, MIOpenHandleDeleter()) {
        CHECK_HIP_ERROR(hipStreamCreate(&stream_));
        CHECK_MIOPEN_ERROR(miopenCreateWithStream(handle_.get(), stream_));
    }
    ~MIOpenHandle()  { 
        CHECK_HIP_ERROR(hipStreamDestroy(stream_));
    }

    miopenHandle_t handle() const { return *handle_; };
};

template<typename T>
class TensorDescriptorNd {
    std::shared_ptr<miopenTensorDescriptor_t> desc_;

    struct TensorDescriptorNdDeleter {
        void operator()(miopenTensorDescriptor_t * desc) {
            miopenDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:

    TensorDescriptorNd(const std::vector<int>& dim,
                       const std::vector<int>& stride) {
        miopenDataType_t type;
        if (std::is_same<T, float>::value)
            type = miopenFloat;
        else
            throw std::runtime_error("Unknown type");

        miopenTensorDescriptor_t * desc = new miopenTensorDescriptor_t;

        CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(desc));
        CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(*desc, type, dim[0], dim[1], dim[2], dim[3]));

        desc_.reset(desc, TensorDescriptorNdDeleter());
    }

    miopenTensorDescriptor_t desc() const { return *desc_; }

};

template<typename T>
class TensorDescriptorNdArray {
    std::shared_ptr<miopenTensorDescriptor_t> desc_array_;

    struct ArrayDeleter {
        int num_;
        ArrayDeleter(int num) : num_(num) {}

        void operator()(miopenTensorDescriptor_t *desc_array) {
            for (int i = 0; i < num_; ++i) {
                miopenDestroyTensorDescriptor(desc_array[i]);
            }

            delete[] desc_array;
        }
    };

    public:

    TensorDescriptorNdArray(std::vector<int> dim,
                            std::vector<int> stride,
                            int num) {
        miopenDataType_t type;
        if (std::is_same<T, float>::value)
            type = miopenFloat;
        else
            throw std::runtime_error("Unknown type");

        miopenTensorDescriptor_t * desc_array = new miopenTensorDescriptor_t[num];

        for (int i = 0; i < num; ++i) {
            CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&desc_array[i]));
            CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(desc_array[i], type,
                                          dim[0], dim[1], dim[2], dim[3]));
        }

        desc_array_.reset(desc_array, ArrayDeleter(num));
    }

    miopenTensorDescriptor_t * ptr() const { return desc_array_.get(); }
};

template<typename T>
class FilterDescriptorNd {
    std::shared_ptr<miopenTensorDescriptor_t> desc_;

    struct FilterDescriptorNdDeleter {
        void operator()(miopenTensorDescriptor_t * desc) {
            miopenDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:

    FilterDescriptorNd() {}

    FilterDescriptorNd(const std::vector<int> dim) {
        miopenDataType_t type;
        if (std::is_same<T, float>::value)
            type = miopenFloat;
        else
            throw std::runtime_error("Unknown type");

        miopenTensorDescriptor_t * desc = new miopenTensorDescriptor_t;
        CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(desc));
        CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(*desc, type, dim[0], dim[1], dim[2], dim[3]));

        desc_.reset(desc, FilterDescriptorNdDeleter());
    }

    miopenTensorDescriptor_t desc() { return *desc_; }
};

template<typename T>
class TensorDescriptor4d {
    std::shared_ptr<miopenTensorDescriptor_t> desc_;

    struct TensorDescriptor4dDeleter {
        void operator()(miopenTensorDescriptor_t * desc) {
            miopenDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:

    TensorDescriptor4d() {}
    TensorDescriptor4d(const int n, const int c, const int h, const int w) {
        miopenDataType_t type;
        if (std::is_same<T, float>::value)
            type = miopenFloat;
        else
            throw std::runtime_error("Unknown type");

        miopenTensorDescriptor_t * desc = new miopenTensorDescriptor_t;
        CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(desc));
        CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(*desc,
                                                     type,
                                                     n,
                                                     c,
                                                     h,
                                                     w));

        desc_.reset(desc, TensorDescriptor4dDeleter());
    }

    miopenTensorDescriptor_t desc() const { return *desc_; }

};

template<typename T>
class FilterDescriptor4d {
    std::shared_ptr<miopenTensorDescriptor_t> desc_;

    struct FilterDescriptor4dDeleter {
        void operator()(miopenTensorDescriptor_t * desc) {
            miopenDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:
    FilterDescriptor4d(int k, int c, int h, int w) {
        miopenDataType_t type;
        if (std::is_same<T, float>::value)
            type = miopenFloat;
        else
            throw std::runtime_error("Unknown type");

        miopenTensorDescriptor_t * desc = new miopenTensorDescriptor_t;
        CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(desc));
        CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(*desc, type, k, c, h, w));

        desc_.reset(desc, FilterDescriptor4dDeleter());
    }

    miopenTensorDescriptor_t desc() const { return *desc_; }

};

class ConvolutionDescriptor {
    std::shared_ptr<miopenConvolutionDescriptor_t> desc_;

    struct ConvolutionDescriptorDeleter {
        void operator()(miopenConvolutionDescriptor_t * desc) {
            miopenDestroyConvolutionDescriptor(*desc);
            delete desc;
        }
    };
public:


    ConvolutionDescriptor(int pad_h, int pad_w, int hstride, int wstride) :
        desc_(new miopenConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {

        CHECK_MIOPEN_ERROR(miopenCreateConvolutionDescriptor(desc_.get()));
        CHECK_MIOPEN_ERROR(miopenInitConvolutionDescriptor(*desc_,
                                                          miopenConvolution,
                                                          pad_h,
                                                          pad_w,
                                                          hstride,
                                                          wstride,
                                                          1,
                                                          1));
    }

    miopenConvolutionDescriptor_t desc() const { return *desc_; };

};


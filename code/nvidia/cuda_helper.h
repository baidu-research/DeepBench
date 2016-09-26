#pragma once

#include <sstream>

#include <cuda.h>

void throw_cuda_error(cudaError_t ret, int line, const char* filename) {
    if (ret != cudaSuccess) {
        std::stringstream ss;
        ss << "Cuda failure: " << cudaGetErrorString(ret) <<
            " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUDA_ERROR(ret) throw_cuda_error(ret, __LINE__, __FILE__)



#pragma once

#include <sstream>

#include <hip/hip_runtime_api.h>

void throw_hip_error(hipError_t ret, int line, const char* filename) {
    if (ret != hipSuccess) {
        std::stringstream ss;
        ss << "HIP failure: " << hipGetErrorString(ret) <<
            " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_HIP_ERROR(ret) throw_hip_error(ret, __LINE__, __FILE__)

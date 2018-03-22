#pragma once

#include "rccl.h"

// Helper function to throw std::runtime_error on rccl failures.
void throw_rccl_error(rcclResult_t ret, int rank, int line, const char* filename) {
    if (ret != rcclSuccess) {
        std::stringstream ss;
        ss << "RCCL failure: " << rcclGetErrorString(ret) <<
            " in " << filename << " at line: " << line << " rank: " << rank << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_RCCL_ERROR(ret, rank) throw_rccl_error(ret, rank, __LINE__, __FILE__)


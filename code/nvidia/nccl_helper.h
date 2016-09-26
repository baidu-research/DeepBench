#pragma once

#include <nccl.h>

// Helper function to throw std::runtime_error on nccl failures.
void throw_nccl_error(ncclResult_t ret, int rank, int line, const char* filename) {
    if (ret != ncclSuccess) {
        std::stringstream ss;
        ss << "NCCL failure: " << ncclGetErrorString(ret) <<
            " in " << filename << " at line: " << line << " rank: " << rank << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_NCCL_ERROR(ret, rank) throw_nccl_error(ret, rank, __LINE__, __FILE__)


#include <iomanip>
#include <chrono>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "nccl_helper.h"
#include "cuda_helper.h"
#include "all_reduce_problems.h"

int all_reduce(int t_size, cudaStream_t * streams, ncclComm_t * comms, int numGpus, int numRepeats) {

    float ** send_buff = (float **)malloc(numGpus * sizeof(float *));
    float ** recv_buff = (float **)malloc(numGpus * sizeof(float *));


    for (int i = 0; i < numGpus; i++) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaMalloc(send_buff+i, t_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(recv_buff+i, t_size * sizeof(float)));

        thrust::fill(thrust::device_ptr<float>(send_buff[i]),
                     thrust::device_ptr<float>(send_buff[i] + t_size), i);
        thrust::fill(thrust::device_ptr<float>(recv_buff[i]),
                     thrust::device_ptr<float>(recv_buff[i] + t_size), 0.f);

    }

    for (int i = 0; i < numGpus; i++) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        cudaStreamSynchronize(streams[i]);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; i++) {
        #if NCCL_MAJOR >= 2
        ncclGroupStart();
        #endif

        for (int j = 0; j < numGpus; j++) {
            //CHECK_CUDA_ERROR(cudaSetDevice(j));
            CHECK_NCCL_ERROR(ncclAllReduce((void *) (send_buff[j]),
                                           (void *) (recv_buff[j]),
                                           t_size,
                                           ncclFloat,
                                           ncclSum,
                                           comms[j],
                                           streams[j]), 0);
        }

        #if NCCL_MAJOR >= 2
        ncclGroupEnd();
        #endif

        for (int j = 0; j < numGpus; j++) {
            CHECK_CUDA_ERROR(cudaSetDevice(j));
            cudaStreamSynchronize(streams[j]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    int time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);


    for (int i = 0; i < numGpus; i++) {
        cudaFree(send_buff[i]);
        cudaFree(recv_buff[i]);
    }

    free(send_buff);
    free(recv_buff);

    return time;

}


int main(int argc, char  **argv) {

    cudaFree(0);
    int nVis;

    int numGpus;

    CHECK_CUDA_ERROR(cudaGetDeviceCount(&nVis));

    if (argc > 1) {
        numGpus = atoi(argv[1]);
    } else {
        throw std::runtime_error("Must specify number of GPUs!");
    }

    if (numGpus > nVis) {
        std::stringstream ss;
        ss << "Number of Gpus Requested: " << numGpus << std::endl;
        ss << "Number of devices visible: " << nVis << std::endl;
        ss << "Number of Gpus requested cannot be more than visible devices" << std::endl;
        throw std::runtime_error(ss.str());
    }

    // Initialize curand_gen and set appropriate seed.
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::vector<int> devList;
    for (int i = 0; i < numGpus; i++) 
        devList.push_back(i);

    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*numGpus);
    CHECK_NCCL_ERROR(ncclCommInitAll(comms, numGpus, devList.data()), 0);

    cudaStream_t * streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*numGpus);
    for (int i = 0; i < numGpus; i++) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaStreamCreate(streams+i));
    }

    std::cout << " NCCL AllReduce " << std::endl;
    std::cout << " Num Ranks: " << numGpus << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    # of floats    bytes transferred    Time (msec)   " << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "-" << std::endl;
    std::cout << std::setfill(' ');


    int64_t* sizes = all_reduce_kernels_size;
    int64_t* numRepeats = all_reduce_kernels_repeat;

    for (int kernel_pos = 0; kernel_pos < _NUMBER_OF_KERNELS_; kernel_pos++) {
        auto t_size = sizes[kernel_pos];
        int time  = all_reduce(t_size, streams, comms, numGpus, numRepeats[kernel_pos]);
        float time_ms = time/1000.0;
        std::cout << std::setw(15) << t_size << std::setw(15) << t_size * 4 << std::setw(20) << time_ms << std::endl;
    }

    for (int i = 0; i < numGpus; i++) {
        ncclCommDestroy(comms[i]);
    }

    free(streams);
    free(comms);

}

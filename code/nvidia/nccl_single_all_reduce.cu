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
        for (int i = 0; i < numGpus; i++) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_NCCL_ERROR(ncclAllReduce((void *) (send_buff[i]),
                                           (void *) (recv_buff[i]),
                                           t_size,
                                           ncclFloat,
                                           ncclSum,
                                           comms[i],
                                           streams[i]), 0);
        }

        for (int i = 0; i < numGpus; i++) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            cudaStreamSynchronize(streams[i]);
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

    int numRepeats = 1000;
    int numGpus;

    CHECK_CUDA_ERROR(cudaGetDeviceCount(&nVis));

    if (argc > 1) {
        numGpus = atoi(argv[1]);
    } else {
        throw std::runtime_error("Must specify number of GPUs!");
    }

    if (argc > 2) {
        numRepeats = atoi(argv[2]);
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


    std::vector<int> sizes = {100000, 3097600, 4194304, 6553600, 16777217};


    for (auto t_size: sizes) {
        int time  = all_reduce(t_size, streams, comms, numGpus, numRepeats);
        float time_ms = time/1000.0;
        std::cout << std::setw(15) << t_size << std::setw(15) << t_size * 4 << std::setw(20) << time_ms << std::endl;
    }

    for (int i = 0; i < numGpus; i++) {
        ncclCommDestroy(comms[i]);
    }

    free(streams);
    free(comms);

}

#include <iomanip>
#include <chrono>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rccl.h"

#include "rccl_helper.h"
#include "hip_helper.h"


int all_reduce(int t_size, hipStream_t * streams, rcclComm_t * comms, int numGpus, int numRepeats) {

    float ** send_buff = (float **)malloc(numGpus * sizeof(float *));
    float ** recv_buff = (float **)malloc(numGpus * sizeof(float *));


    for (int i = 0; i < numGpus; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));
        CHECK_HIP_ERROR(hipMalloc(send_buff+i, t_size * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(recv_buff+i, t_size * sizeof(float)));

//        thrust::fill(thrust::device_ptr<float>(send_buff[i]),
//                     thrust::device_ptr<float>(send_buff[i] + t_size), i);
//        thrust::fill(thrust::device_ptr<float>(recv_buff[i]),
//                     thrust::device_ptr<float>(recv_buff[i] + t_size), 0.f);

    }

    for (int i = 0; i < numGpus; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));
        hipStreamSynchronize(streams[i]);
    }

    auto start = std::chrono::steady_clock::now();


    for (int i = 0; i < numRepeats; i++) {
        for (int i = 0; i < numGpus; i++) {
            CHECK_HIP_ERROR(hipSetDevice(i));

            CHECK_RCCL_ERROR(rcclAllReduce((void *) (send_buff[i]),
                                           (void *) (recv_buff[i]),
                                           t_size,
                                           rcclFloat,
                                           rcclSum,
                                           comms[i],
                                           streams[i]), 0);

        }

        for (int i = 0; i < numGpus; i++) {
            CHECK_HIP_ERROR(hipSetDevice(i));
            hipStreamSynchronize(streams[i]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    int time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);


    for (int i = 0; i < numGpus; i++) {
        hipFree(send_buff[i]);
        hipFree(recv_buff[i]);
    }

    free(send_buff);
    free(recv_buff);

    return time;

}


int main(int argc, char  **argv) {

    hipFree(0);
    int nVis;

    int numRepeats = 1000;
    int numGpus;

    CHECK_HIP_ERROR(hipGetDeviceCount(&nVis));

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
//    curandGenerator_t curand_gen;
//    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
//    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::vector<int> devList;
    for (int i = 0; i < numGpus; i++) 
        devList.push_back(i);

    rcclComm_t* comms = (rcclComm_t*)malloc(sizeof(rcclComm_t)*numGpus);
    CHECK_RCCL_ERROR(rcclCommInitAll(comms, numGpus, devList.data()), 0);

    hipStream_t * streams = (hipStream_t*)malloc(sizeof(hipStream_t)*numGpus);
    for (int i = 0; i < numGpus; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));
        CHECK_HIP_ERROR(hipStreamCreate(streams+i));
        for(int j = 0; j < numGpus; j++) {
            if(i != j) {
                CHECK_HIP_ERROR(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }

    std::cout << " RCCL AllReduce " << std::endl;
    std::cout << " Num Ranks: " << numGpus << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    # of floats    bytes transferred    Time (msec)   Bandwidth (GB/s)" << std::endl;

    std::cout << std::setfill('-') << std::setw(75) << "-" << std::endl;
    std::cout << std::setfill(' ');


    std::vector<int> sizes = {100000, 3097600, 4194304, 6553600, 16777217};


    for (auto t_size: sizes) 
    {
        int time  = all_reduce(t_size, streams, comms, numGpus, numRepeats);
        float time_ms = time/1000.0;
        std::cout << std::setw(15) << t_size << std::setw(15) << t_size * 4 << std::setw(20) << time_ms << std::setw(20) << t_size*4/time_ms/1e6<<std::endl;
    }

    for (int i = 0; i < numGpus; i++) {
        rcclCommDestroy(comms[i]);
    }

    free(streams);
    free(comms);

}

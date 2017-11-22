#include "collectives.h"

#include <iomanip>
#include <sstream>

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include "all_reduce_problems.h"
#include <chrono>


int main(int argc, char** argv, char** envp) {

    int mpi_size, mpi_rank, mpi_local_rank;

    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
		env_str = std::getenv("SLURM_LOCALID");
    }

    mpi_local_rank = std::stoi(std::string(env_str));

    InitCollectives(mpi_local_rank);
	
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    int64_t* sizes = all_reduce_kernels_size;
    int64_t* numRepeats = all_reduce_kernels_repeat;

    if (mpi_rank == 0) {
        std::cout << " Ring AllReduce " << std::endl;
        std::cout << " Num Ranks: " << mpi_size << std::endl;

        std::cout << std::setfill('-') << std::setw(100) << "-" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "    # of floats    bytes transferred    Avg Time (msec)    Max Time (msec)" << std::endl;

        std::cout << std::setfill('-') << std::setw(100) << "-" << std::endl;
        std::cout << std::setfill(' ');
    }
 
    cudaError_t err;
    for (int kernel_pos = 0; kernel_pos < _NUMBER_OF_KERNELS_; kernel_pos++) {
        auto t_size = sizes[kernel_pos];
        
        float* cpu_data = new float[t_size];
        std::fill_n(cpu_data, t_size, 1.0f);

        float* data;
        err = cudaMalloc(&data, sizeof(float) * t_size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed!"); }

        err = cudaMemcpy(data, cpu_data, sizeof(float) * t_size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed!"); }

        float time_sum = 0;
        for (int i = 0; i < numRepeats[kernel_pos]; i++) {

            float* output;

            auto start = std::chrono::steady_clock::now();
            RingAllreduce(data, t_size, &output);
            auto end = std::chrono::steady_clock::now();
            time_sum += std::chrono::duration<double, std::milli>(end - start).count();            

            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed!"); }
        }

        float time = static_cast<float>(time_sum / numRepeats[kernel_pos]);

        float max_time, avg_time;
        MPI_Reduce(&time, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time, &avg_time, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            avg_time = avg_time/mpi_size;
            std::cout << std::setw(15) << t_size << std::setw(15) << t_size * 4 << std::setw(20) << avg_time << std::setw(20) << max_time << std::endl;
        }
        
        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed!"); }

        delete [] cpu_data;
    }        

    MPI_Finalize();

    return 0;
}

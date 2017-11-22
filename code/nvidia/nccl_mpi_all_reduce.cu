#include <iomanip>
#include <sstream>
#include <chrono>

#include <cuda.h>

#include <mpi.h>

#include "tensor.h"
#include "nccl_helper.h"
#include "all_reduce_problems.h"

int main(int argc, char *argv[]) {

    int size, rank;

    int mpi_local_rank;

    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
		env_str = std::getenv("SLURM_LOCALID");
    }
    mpi_local_rank = std::stoi(std::string(env_str));

    //Initialize MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Set cuda devices
    if (cudaSetDevice(mpi_local_rank) != cudaSuccess) {
        std::stringstream ss;
        ss << "Failed to set cuda device. Rank: " << rank;
        throw std::runtime_error(ss.str());
    }

    //NCCL communicator
    ncclComm_t comm;
    ncclUniqueId commId;

    // NCCL init and set up communicator clique
    CHECK_NCCL_ERROR(ncclGetUniqueId(&commId), rank);
    MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_NCCL_ERROR(ncclCommInitRank(&comm, size, commId, rank), rank);

    // CUDA stream creation
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);


    int64_t* sizes = all_reduce_kernels_size;
    int64_t* numRepeats = all_reduce_kernels_repeat;


    if (rank == 0) {
        std::cout << " NCCL MPI AllReduce " << std::endl;
        std::cout << " Num Ranks: " << size << std::endl;

        std::cout << std::setfill('-') << std::setw(100) << "-" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "    # of floats    bytes transferred    Avg Time (msec)    Max Time (msec)" << std::endl;

        std::cout << std::setfill('-') << std::setw(100) << "-" << std::endl;
        std::cout << std::setfill(' ');

    }
    
    for (int kernel_pos = 0; kernel_pos < _NUMBER_OF_KERNELS_; kernel_pos++) {

        auto t_size = sizes[kernel_pos];
 
        auto data = fill<float>({(int)t_size*size}, rank);

        cudaStreamSynchronize(stream);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < numRepeats[kernel_pos]; i++) {
            CHECK_NCCL_ERROR(ncclAllReduce((void *) data.begin(), 
                                           (void *) (data.begin() + t_size), 
                                           t_size, 
                                           ncclFloat, 
                                           ncclSum, 
                                           comm, 
                                           stream), rank);

            cudaStreamSynchronize(stream);
        }

        auto end = std::chrono::steady_clock::now();
        float time = static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count() / numRepeats[kernel_pos]);

        float max_time, avg_time;
        MPI_Reduce(&time, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time, &avg_time, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            avg_time = avg_time/size;
            std::cout << std::setw(15) << t_size << std::setw(15) << t_size * 4 << std::setw(20) << avg_time << std::setw(20) << max_time << std::endl;
        }
    }

    ncclCommDestroy(comm);
    MPI_Finalize();
}

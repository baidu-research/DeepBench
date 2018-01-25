
CC=mpic++
NVCC=nvcc
ARCH?=sm_52

CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
BAIDU_ALLREDUCE_PATH?=/local/baidu-allreduce
BIN_DIR?=bin
MKDIR=mkdir -p
#BLAS
BLAS_LIBRARY?=cublas
BLAS_PATH?=$(CUDA_LIB64)
#CONV
KERNELS_DIR=../kernels/
COMMA=,
NVCC_ARCH_ARGS=$(foreach a,$(subst $(COMMA), ,$(ARCH)),--generate-code arch=$(patsubst sm_%,compute_%,$(a)),code=$(a))

.PHONY=all ring_all_reduce clean

ring_all_reduce:
	$(MKDIR) $(BIN_DIR)
	$(MPI_PATH)/bin/$(CC) -c -std=c++11 -I $(MPI_INCLUDE_PATH) -I $(BAIDU_ALLREDUCE_PATH) -I $(CUDA_PATH)/include -I $(KERNELS_DIR) -DOMPI_SKIP_MPICXX= ring_all_reduce_mpi.cpp -o $(BIN_DIR)/ring_all_reduce_mpi.o
	$(CUDA_PATH)/bin/$(NVCC) -c -std=c++11 -I $(MPI_INCLUDE_PATH) -I $(BAIDU_ALLREDUCE_PATH) -I $(CUDA_PATH)/include -DOMPI_SKIP_MPICXX= $(BAIDU_ALLREDUCE_PATH)/collectives.cu -o $(BIN_DIR)/collectives.o
	$(MPI_PATH)/bin/$(CC) -o $(BIN_DIR)/ring_all_reduce $(BIN_DIR)/ring_all_reduce_mpi.o $(BIN_DIR)/collectives.o -L$(CUDA_PATH)/lib64 -L$(MPI_PATH)/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX=

clean:
	rm -rf $(BIN_DIR)

rebuild: clean all

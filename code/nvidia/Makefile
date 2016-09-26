
NVCC=nvcc
ARCH=sm_52

CUDA_PATH?=/usr/local/cuda
CUDNN_PATH?=/usr/local/cudnn
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
BIN_DIR?=bin
MKDIR=mkdir -p

.PHONY=all gemm conv rnn all_reduce nccl_single nccl_mpi clean

all: gemm conv rnn all_reduce

gemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) gemm_bench.cu -o $(BIN_DIR)/gemm_bench -I $(CUDA_PATH)/include -L $(CUDA_PATH)/lib64 -lcublas -lcurand -arch=$(ARCH) -std=c++11

conv:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) conv_bench.cu -o $(BIN_DIR)/conv_bench -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_PATH)/lib64 -lcurand -lcudnn -arch=$(ARCH) -std=c++11

rnn:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) rnn_bench.cu -o $(BIN_DIR)/rnn_bench -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_PATH)/lib64 -lcurand -lcudnn -arch=$(ARCH) -std=c++11

all_reduce: nccl_single nccl_mpi

nccl_single:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_single_all_reduce.cu -o $(BIN_DIR)/nccl_single_all_reduce -I $(NCCL_PATH)/build/include/ -I $(CUDNN_PATH)/include/ -L $(NCCL_PATH)/build/lib/ -L $(CUDNN_PATH)/lib64 -lnccl -lcurand -arch=$(ARCH) -std=c++11 

nccl_mpi:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_mpi_all_reduce.cu -o $(BIN_DIR)/nccl_mpi_all_reduce -I $(NCCL_PATH)/build/include/ -I $(CUDNN_PATH)/include/ -I $(MPI_PATH)/include -L $(NCCL_PATH)/build/lib/ -L $(CUDNN_PATH)/lib64 -L $(MPI_PATH)/lib -lnccl -lcurand -lmpi -arch=$(ARCH) -std=c++11 

clean:
	rm -rf $(BIN_DIR)

rebuild: clean all

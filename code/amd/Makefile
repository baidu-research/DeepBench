
HIPCC=hipcc

HIP_PATH?=/opt/rocm/hip
HIP_LIB64=$(HIP_PATH)/lib
MIOPEN_PATH?=/opt/rocm
BIN_DIR?=bin
MKDIR=mkdir -p

#BLAS
ROCBLAS_PATH=/opt/rocm/rocblas
ROCBLAS_LIB=rocblas-hcc

#CONV
CONV_LIBRARY?=MIOpen
CONV_PATH?=$
TINYGEMM_PATH=/opt/rocm/tgemm
TINYGEMM_LIB=tinygemm
.PHONY=all conv clean

all: conv gemm

OPT=-O3

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIP_PATH)/bin/$(HIPCC) conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(HIP_PATH)/include -I$(MIOPEN_PATH)/include -L$(MIOPEN_PATH)/lib -L$(HIP_LIB64) -l$(CONV_LIBRARY) --amdgpu-target=gfx803 --amdgpu-target=gfx900 $(OPT) -std=c++11 -DMIOPEN_BACKEND_HIP=1

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIP_PATH)/bin/$(HIPCC) gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(HIP_PATH)/include -I$(ROCBLAS_PATH)/include/ -L$(ROCBLAS_PATH)/lib -L$(HIP_LIB64) -l$(ROCBLAS_LIB) --amdgpu-target=gfx803 --amdgpu-target=gfx900 $(OPT) -std=c++11

clean:
	rm -rf $(BIN_DIR)


rebuild: clean all

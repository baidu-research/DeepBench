SOURCE_DIR?=.

HIPCC=/opt/rocm/bin/hipcc
ROCM_PATH=/opt/rocm

BIN_DIR?=bin
MKDIR=mkdir -p

#BLAS
ROCBLAS_LIB=rocblas

#CONV
CONV_LIBRARY?=MIOpen
CONV_PATH?=$
.PHONY=all conv clean

INC_DIR=${SOURCE_DIR}/../kernels

all: conv gemm 

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3
TARGETS=

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(ROCM_PATH)/include -I$(INC_DIR) -l$(CONV_LIBRARY) $(OPT) -std=c++11 --amdgpu-target=gfx900

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(ROCM_PATH)/include -I$(INC_DIR) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

clean:
	rm -rf $(BIN_DIR)


rebuild: clean all

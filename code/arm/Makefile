all: gemm_cblas gemm conv sparse

BIN_DIR?=bin
EIGEN_PATH?=/usr/local/eigen
MKDIR=mkdir -p

ARM_COMPUTE_LIB_PATH?=/usr/local/ARMComputeLib/
ARM_COMPUTE_INCLUDE_PATH?=$(ARM_COMPUTE_LIB_PATH)/include/
ARM_COMPUTE_LIB=$(ARM_COMPUTE_LIB_PATH)/build/

GEMMLOWP_INCLUDE_PATH?=/usr/local/gemmlowp
GEMMLOWP_PATH=$(GEMMLOWP_INCLUDE_PATH)/../

ARMPL_PATH?=/usr/local/armpl/
ARMPL_INCLUDE_PATH=$(ARMPL_PATH)/include/
ARMPL_LIB_PATH=$(ARMPL_PATH)/lib/
KERNELS_DIR=../kernels/

gemm_cblas: cblas_bench.cpp
	$(MKDIR) $(BIN_DIR)
	g++-7 -O3 -fopenmp --std=c++11 -I $(KERNELS_DIR) -I $(ARMPL_INCLUDE_PATH) -L $(ARMPL_LIB_PATH) -o bin/gemm_cblas_bench $< -larmpl -lgfortran

gemm: gemm_bench.cc
	$(MKDIR) $(BIN_DIR)
	g++ -O3 --std=c++11 -I $(KERNELS_DIR) -I $(GEMMLOWP_PATH) -o bin/gemm_bench $< -lpthread

sparse: sparse_bench.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -std=c++11  -o bin/sparse_bench $< \
		-I $(EIGEN_PATH) -I $(KERNELS_DIR)

conv: convolution.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -std=c++11  -I $(ARM_COMPUTE_INCLUDE_PATH) -I $(ARM_COMPUTE_LIB_PATH) -I $(KERNELS_DIR) -L $(ARM_COMPUTE_LIB) $< -o bin/conv_bench -larm_compute_core -larm_compute

clean:
	rm -rf $(BIN_DIR)

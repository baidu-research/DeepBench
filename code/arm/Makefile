all: gemm_armpl gemm_oblas gemm_armcl gemm conv sparse

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

OPENBLAS_PATH?=/usr/local/openblas/
OPENBLAS_INCLUDE_PATH=$(OPENBLAS_PATH)/include/
OPENBLAS_LIB_PATH=$(OPENBLAS_PATH)/lib/

KERNELS_DIR=../kernels/

gemm_armpl: cblas_bench.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -fopenmp --std=c++11 -I $(KERNELS_DIR) -I $(ARMPL_INCLUDE_PATH) -L $(ARMPL_LIB_PATH) -DUSE_ARMPL -o bin/gemm_armpl_bench $< -larmpl_mp -lgfortran

gemm_oblas: cblas_bench.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -fopenmp --std=c++11 -I $(KERNELS_DIR) -I $(OPENBLAS_INCLUDE_PATH) -L $(OPENBLAS_LIB_PATH) -DUSE_OPENBLAS -o bin/gemm_oblas_bench $< -lopenblas

gemm_armcl: armcl_bench.cpp 
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -std=c++11  -I $(ARM_COMPUTE_INCLUDE_PATH) -I $(ARM_COMPUTE_LIB_PATH) -I $(KERNELS_DIR) -L $(ARM_COMPUTE_LIB) $< -o bin/gemm_armcl_bench -larm_compute_core -larm_compute

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


CC=gcc
CC_FLAGS= -c -O2 -pthread -Wall -march=native

MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
CUDA_PATH?=/usr/local/cuda
MKDIR=mkdir -p
BIN_DIR?=bin
KERNELS_DIR=../kernels/

.PHONY= coll all_reduce build clean

all: build

build: coll allreduce
	$(CC) -o $(BIN_DIR)/osu_allreduce -pthread -Wl,--enable-new-dtags -Wl,-rpath=$(CUDA_PATH)/lib64 -Wl,-rpath=$(MPI_PATH)/lib $(BIN_DIR)/osu_allreduce.o $(BIN_DIR)/osu_coll.o -L$(CUDA_PATH)/lib64 -L$(MPI_PATH)/lib -lstdc++ -lmpi_cxx -lmpi -lcuda

coll:
	$(MKDIR) $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/osu_coll.o $(CC_FLAGS) -I$(CUDA_PATH)/include -I$(MPI_INCLUDE_PATH) osu_coll.c

allreduce:
	$(CC) -o $(BIN_DIR)/osu_allreduce.o $(CC_FLAGS) -I $(KERNELS_DIR) -I$(CUDA_PATH)/include -I$(MPI_INCLUDE_PATH) osu_allreduce.c

clean:
	rm -rf $(BIN_DIR)

rebuild: clean build

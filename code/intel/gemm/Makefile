#******************************************************************************
# Copyright 2016 Intel Corporation
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#     http://www.apache.org/licenses/LICENSE-2.0
#     
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

CC       =  icc
CFLAGS	 =  -O2 -Wall -I$(MKLROOT)/include -I../../kernels -fopenmp -std=c++11

EXTRALIB = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -L$(MKLROOT)/lib/intel64/ -liomp5 -lpthread -lm -ldl

OPENBLAS_PATH?=/usr/local/openblas

OPENBLAS_FLAGS = -O2 -Wall -I$(OPENBLAS_PATH)/include -I../../kernels -fopenmp -std=c++11 -DUSE_OPENBLAS
OPENBLAS_LIBS = -L$(OPENBLAS_PATH)/lib -DUSE_OPENBLAS -lpthread -lm -ldl -lopenblas

all : sbench sbench_pack ibench_s8u8s32 sbench_oblas

ibench_s8u8s32 : ibench_s8u8s32.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

ibench_s8u8s32.o : bench.cpp ../../kernels/gemm_problems.h
	$(CC) $(CFLAGS) -DIGEMM_S8U8S32 -c -o $@ $<

sbench_pack : sbench_pack.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

sbench_pack.o : bench.cpp ../../kernels/gemm_problems.h
	$(CC) $(CFLAGS) -DPACKED_API -c -o $@ $<

sbench.o: bench.cpp ../../kernels/gemm_problems.h
	$(CC) $(CFLAGS) -c -o $@ $<

sbench : sbench.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

sbench_oblas.o : bench.cpp ../../kernels/gemm_problems.h
	$(CC) $(OPENBLAS_FLAGS) -c -o $@ $<

sbench_oblas: sbench_oblas.o
	$(CC) $(OPENBLAS_FLAGS) $^ $(OPENBLAS_LIBS) -o $@

clean :
	rm -f *.o sbench sbench_pack ibench_s8u8s32

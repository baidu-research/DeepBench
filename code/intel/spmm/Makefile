#******************************************************************************
# Copyright 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

ifeq ($(MKLROOT),)
    $(error MKLROOT is empty. Please install Intel MKL 2017 or newer)
endif

EXTRACXXFLAGS = -I$(MKLROOT)/include
EXTRALIB = -L$(MKLROOT)/lib/intel64/ -lmkl_rt \
		   -Wl,-rpath,$(MKLROOT)/lib/intel64 -liomp5 -lpthread -lm -ldl

ifeq ($(DEBUG), 1)
OPTFLAGS = -O0 -g
else
OPTFLAGS = -O3 -fopenmp
endif

CXX      = icpc
CXXFLAGS = -Wall -std=c++11 $(OPTFLAGS) $(EXTRACXXFLAGS) -I../../kernels
LFLAGS   = -lrt
OBJS     = spmm_bench.o

all : spmm_bench

spmm_bench : spmm_bench.o
	$(CXX) $(CXXFLAGS) $(LFLAGS) $^  $(EXTRALIB)  -o $@

spmm_bench.o : spmm_bench.cpp ../../kernels/sparse_gemm_problems.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean :
	rm -f *.o spmm_bench.o spmm_bench


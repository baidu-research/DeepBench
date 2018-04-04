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

CONVLIB ?= MKL

ifeq ($(CONVLIB),MKL)
ifeq ($(MKLROOT),)
    $(error MKLROOT is empty. Please install Intel MKL 2017 or newer, or \
        download an Intel MKLML package from \
        https://github.com/intel/caffe/releases.)
endif
PREP_MKL = $(shell sh ../../externals/prepare_mkl.sh)
MKLROOT  = $(word 1, $(PREP_MKL))
MKLLIB   = $(word 2, $(PREP_MKL))
EXTRACXXFLAGS = -I$(MKLROOT)/include -DUSE_MKL
ifeq ($(MKLLIB), mklml_intel)
    EXTRALIB = -L$(MKLROOT)/lib -lmklml_intel
endif
ifeq ($(MKLLIB), mkl_rt)
    EXTRALIB = -L$(MKLROOT)/lib/intel64/ -lmkl_rt \
			   -Wl,-rpath,$(MKLROOT)/lib/intel64
endif
EXTRALIB += -liomp5 -lpthread -lm -ldl
endif

ifeq ($(CONVLIB),MKLDNN)
ifeq ($(MKLDNNROOT),)
    $(error MKLDNNROOT is empty. Please build MKLDNN from \
        https://github.com/01org/mkl-dnn and point MKLDNNROOT \
        to the install directory.)
endif
EXTRACXXFLAGS = -I$(MKLDNNROOT)/include -DUSE_MKLDNN
EXTRALIB = -L$(MKLDNNROOT)/lib -lmkldnn -Wl,-rpath,$(MKLDNNROOT)/lib
endif

ifeq ($(DEBUG), 1)
OPTFLAGS = -O0 -g
else
OPTFLAGS = -O3
endif

CXX      = icpc
CXXFLAGS = -Wall -std=c++11 $(OPTFLAGS) $(EXTRACXXFLAGS) -fopenmp -I../../../kernels
LFLAGS   = -lrt
OBJS     = std_conv_bench.o

all : std_conv_bench

std_conv_bench : std_conv_bench.o
	$(CXX) $(CXXFLAGS) $(LFLAGS) $^  $(EXTRALIB)  -o $@

std_conv_bench.o : std_conv_bench.cpp ../../../kernels/conv_problems.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean :
	rm -f *.o std_conv_bench.o std_conv_bench


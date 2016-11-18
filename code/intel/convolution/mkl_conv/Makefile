#******************************************************************************
# Copyright 2016 Intel Corporation
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
$(error MKLROOT must be defined. It's possible to use Intel MKL 2017 Gold and higher or Intel MKLML package from IntelCaffe: https://github.com/intel/caffe/releases)
endif

RETURN_STRING=$(shell sh ../../externals/prepare_mkl.sh)

MKLROOT=$(firstword $(RETURN_STRING))
MKLLIB=$(word 2, $(RETURN_STRING))

tmp=$(shell echoi "Here")
tmp=$(shell echo $MKLLIB)

CC       =  icc
CFLAGS   =  -O3  -Wall -I$(MKLROOT)/include -qopenmp -std=c++11
OBJS     =  std_conv_bench.o

ifeq ($(MKLLIB), mklml_intel)
EXTRALIB = -L$(MKLROOT)/lib -lmklml_intel -liomp5 -lpthread -lm -ldl
endif

ifeq ($(MKLLIB), mkl_rt)
EXTRALIB = -L$(MKLROOT)/lib/intel64/ -lmkl_rt -liomp5 -lpthread -lm -ldl
endif

all : std_conv_bench

std_conv_bench : std_conv_bench.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

std_conv_bench.o : std_conv_bench.cpp input.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o std_conv_bench.o std_conv_bench


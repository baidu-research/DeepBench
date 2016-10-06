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


CC       =  icc
CFLAGS   =  -O3 -Wall -I$(MKLROOT)/include -qopenmp -std=c++11
OBJS     =  std_conv_bench.o

EXTRALIB = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

all : std_conv_bench

std_conv_bench : std_conv_bench.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

std_conv_bench.o : std_conv_bench.cpp input.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o std_conv_bench.o std_conv_bench

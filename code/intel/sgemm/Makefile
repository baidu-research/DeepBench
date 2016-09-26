#*******************************************************************************
#* Copyright 2016 Intel Corporation All Rights Reserved.
#*
#* The source code,  information  and material  ("Material") contained  herein is
#* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
#* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
#* Material  contains  proprietary  information  of  Intel or  its suppliers  and
#* licensors.  The Material is protected by  worldwide copyright  laws and treaty
#* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
#* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
#* in any way without Intel's prior express written permission.  No license under
#* any patent,  copyright or other  intellectual property rights  in the Material
#* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
#* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
#* property rights must be express and approved by Intel in writing.
#*
#* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
#* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
#* suppliers or licensors in any way.
#*******************************************************************************

CC       =  icc
CFLAGS	 =  -O2 -Wall -I$(MKLROOT)/include -openmp
OBJS     =  sbench.o

EXTRALIB = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

all : sbench sbench_pack

sbench_pack : sbench_pack.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

sbench_pack.o : sbench.c input.h
	$(CC) $(CFLAGS) -DPACKED_API -c -o $@ $<


sbench : sbench.o
	$(CC) $(CFLAGS) $^  $(EXTRALIB)  -o $@ 

sbench.o : sbench.c input.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o sbench  sbench_pack

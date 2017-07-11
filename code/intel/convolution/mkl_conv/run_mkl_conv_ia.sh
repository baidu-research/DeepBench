#!/bin/bash

#******************************************************************************
# Copyright 2016-2017 Intel Corporation
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
#*******************************************************************************

echo "Please source appropriate versions of Intel Compiler (ICC) and " \
    "Intel MKL, and build MKL-DNN 0.9 or later"
# source <ICC_INSTALDIR>
# source <MKL_INSTALDIR>
# export DNNROOT=<MKLDNN_INSTALL_DIR>

export LD_LIBRARY_PATH=$DNNROOT/lib:$LD_LIBRARY_PATH

export KMP_HW_SUBSET=1T
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=$(lscpu | grep 'Core(s) per socket' | awk '{print $NF}')

make clean all CONVLIB=MKLDNN || \
    { echo "*** ERROR: make failed"; exit 1; }

if lscpu | grep Flags | grep -qs avx512dq; then
    ./run_mkl_conv_ia_SKX.sh
elif lscpu | grep Flags | grep -qa avx512f; then
    ./run_mkl_conv_ia_KNL.sh
else
    ./run_mkl_conv_ia_generic.sh
fi

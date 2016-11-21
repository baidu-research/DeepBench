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
#*******************************************************************************

#!/bin/bash
export KMP_PLACE_THREADS=1T
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=64

# source ICC
echo "Please source appropriate versions of Intel Compiler (ICC), Intel MPI and  Intel MKL !"
echo "------------------------"
echo " libxsmm Convolution - "
echo "--------------"
echo " "

rm -rf libxsmm
git clone https://github.com/hfp/libxsmm.git
cd libxsmm

echo "Building libxsmm..."
make realclean
make -j MAKE_PARALLEL=1 &> /dev/null

cd samples/dnn
pwd
make realclean
make -j MAKE_PARALLEL=1 &> /dev/null

sh run_deepbench.sh

cd ../../../

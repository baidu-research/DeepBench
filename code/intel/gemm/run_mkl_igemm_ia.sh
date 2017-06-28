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

#!/bin/bash

echo "Please source appropriate versions of Intel Compiler (ICC), Intel MPI and  Intel MKL !"
#source <ICC_INSTALDIR>
#source <MKL_INSTALDIR>
#source <IMPI_INSTALDIR>

# Intel Xeon Phi Settings
export KMP_PLACE_THREADS=1T
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=66

echo " IGEMM benchmark"
make clean &> /dev/null;
make ibench_s8u8s32 &> /dev/null;
echo " "
echo "------------------------"
echo " GEMM_S8U8S32 - "
echo "--------------"
echo " "
numactl -m 1 ./ibench_s8u8s32

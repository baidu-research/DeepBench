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
make -j &> /dev/null

cd samples/dnn
pwd
make realclean
make -j &> /dev/null

cd ../../
pwd
BIN=samples/dnn/layer_example_f32
ITERS=1000
NUMA=1

# @TODO this handles cache and quad cluster mode, but not SNC4
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA}"
  fi
fi

# Usage: ./layer_example_f32 iters inpWidth inpHeight minibatch nIfm nOfm kw kh pad stride splits

${NUMACTL} ${BIN} ${ITERS}  700 161 4  1   32  5  20  0  2  1
${NUMACTL} ${BIN} ${ITERS}  700 161 8  1   32  5  20  0  2  1
${NUMACTL} ${BIN} ${ITERS}  700 161 16 1   32  5  20  0  2  1
${NUMACTL} ${BIN} ${ITERS}  700 161 32 1   32  5  20  0  2  1
${NUMACTL} ${BIN} ${ITERS}  341 79  4  32  32  5  10  0  2  1
${NUMACTL} ${BIN} ${ITERS}  341 79  8  32  32  5  10  0  2  1
${NUMACTL} ${BIN} ${ITERS}  341 79  16 32  32  5  10  0  2  1
${NUMACTL} ${BIN} ${ITERS}  341 79  32 32  32  5  10  0  2  1
${NUMACTL} ${BIN} ${ITERS}  480 48  16 1   16  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  240 24  16 16  32  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  120 12  16 32  64  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  60  6   16 64  128 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  108 108 8  3   64  3  3   1  2  1
${NUMACTL} ${BIN} ${ITERS}  54  54  8  64  64  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  27  27  8  128 128 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  14  14  8  128 256 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  7   7   8  256 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  224 224 8  3   64  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  112 112 8  64  128 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  56  56  8  128 256 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  28  28  8  256 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  14  14  8  512 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  7   7   8  512 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  224 224 16 3   64  3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  112 112 16 64  128 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  56  56  16 128 256 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  28  28  16 256 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  14  14  16 512 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  7   7   16 512 512 3  3   1  1  1
${NUMACTL} ${BIN} ${ITERS}  224 224 16 3   64  7  7   3  2  1
${NUMACTL} ${BIN} ${ITERS}  28  28  16 192 32  5  5   2  1  1
${NUMACTL} ${BIN} ${ITERS}  28  28  16 192 64  1  1   0  1  1
${NUMACTL} ${BIN} ${ITERS}  14  14  16 512 48  5  5   2  1  1
${NUMACTL} ${BIN} ${ITERS}  14  14  16 512 192 1  1   0  1  1
${NUMACTL} ${BIN} ${ITERS}  7   7   16 832 256 1  1   0  1  1
${NUMACTL} ${BIN} ${ITERS}  7   7   16 832 128 5  5   2  1  1

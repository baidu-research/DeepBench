#*******************************************************************************
# * Copyright 2016 Intel Corporation All Rights Reserved.
# *
# * The source code,  information  and material  ("Material") contained  herein is
# * owned by Intel Corporation or its  suppliers or licensors,  and  title to such
# * Material remains with Intel  Corporation or its  suppliers or  licensors.  The
# * Material  contains  proprietary  information  of  Intel or  its suppliers  and
# * licensors.  The Material is protected by  worldwide copyright  laws and treaty
# * provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
# * modified, published,  uploaded, posted, transmitted,  distributed or disclosed
# * in any way without Intel's prior express written permission.  No license under
# * any patent,  copyright or other  intellectual property rights  in the Material
# * is granted to  or  conferred  upon  you,  either   expressly,  by implication,
# * inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
# * property rights must be express and approved by Intel in writing.
# *
# * Unless otherwise agreed by Intel in writing,  you may not remove or alter this
# * notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
# * suppliers or licensors in any way.
# *******************************************************************************/

#!/bin/bash
export KMP_PLACE_THREADS=1T
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=66

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
make &> /dev/null

cd samples/dnn
make realclean
make &> /dev/null

cd ../../
BIN=samples/dnn/fwd_layer_example
ITERS=1000
NUMA=1

# Usage: ./fwd_layer_example iters inpWidth inpHeight minibatch nIfm nOfm kw kh pad stride splits

numactl --membind=${NUMA} ${BIN} ${ITERS}  700 161 4  1   32  5  20  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  700 161 8  1   32  5  20  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  700 161 16 1   32  5  20  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  700 161 32 1   32  5  20  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  341 79  4  32  32  5  10  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  341 79  8  32  32  5  10  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  341 79  16 32  32  5  10  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  341 79  32 32  32  5  10  0  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  480 48  16 1   16  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  240 24  16 16  32  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  120 12  16 32  64  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  60  6   16 64  128 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  108 108 8  3   64  3  3   1  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  54  54  8  64  64  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  27  27  8  128 128 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  14  14  8  128 256 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  7   7   8  256 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  224 224 8  3   64  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  112 112 8  64  128 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  56  56  8  128 256 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  28  28  8  256 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  14  14  8  512 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  7   7   8  512 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  224 224 16 3   64  3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  112 112 16 64  128 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  56  56  16 128 256 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  28  28  16 256 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  14  14  16 512 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  7   7   16 512 512 3  3   1  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  224 224 16 3   64  7  7   3  2  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  28  28  16 192 32  5  5   2  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  28  28  16 192 64  1  1   0  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  14  14  16 512 48  5  5   2  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  14  14  16 512 192 1  1   0  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  7   7   16 832 256 1  1   0  1  1
numactl --membind=${NUMA} ${BIN} ${ITERS}  7   7   16 832 128 5  5   2  1  1

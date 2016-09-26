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

#echo "Please source appropriate versions of Intel Compiler (ICC), Intel MPI and  Intel MKL !"
#source <ICC_INSTALDIR>
#source <MKL_INSTALDIR>
#source <IMPI_INSTALDIR>

export KMP_PLACE_THREADS=1T
export KMP_AFFINITY=compact,granularity=fine
export OMP_NUM_THREADS=66

echo " "
echo " Convolution benchmark"
cd mkl_conv
pwd
sh ./run_mkl_conv_ia.sh
echo " "
cd ../libxsmm_conv
pwd
echo " "
sh ./run_libxsmm_conv_ia.sh
cd ..

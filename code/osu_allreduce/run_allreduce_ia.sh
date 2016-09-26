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

#!/bin/sh

function run_osu {

numnodes=$1
bin=$2
ppncpu=$3
hostfile=$4
fabric=$5

numthreads=1

if [ "$fabric" != "openmpi" ]; then
  if [ "$fabric" == "" ] || [ "$fabric" == "ofi" ]; then
    # export PSM2_MQ_RNDV_HFI_WINDOW=4194304
    # export PSM2_MQ_EAGER_SDMA_SZ=65536
    # export PSM2_MQ_RNDV_HFI_THRESH=200000
    export PSM2_IDENTIFY=1
    export PSM2_RCVTHREAD=0
    export PSM2_SHAREDCONTEXTS=0
    if [ "$fabric" == "ofi" ]; then
      export I_MPI_FABRICS=ofi
    else
      export I_MPI_FABRICS=tmi
      export I_MPI_TMI_PROVIDER=psm2
    fi
  elif [ "$fabric" == "dapl" ]; then
    export I_MPI_FABRICS=dapl
    export I_MPI_DAPL_PROVIDER=ofa-v2-hfi1_0-1
  else
    echo "Unknown I_MPI_FABRICS"
    exit
  fi
  export HFI_NO_CPUAFFINITY=1
  export IPATH_NO_CPUAFFINITY=1
  export I_MPI_FALLBACK=0
  export I_MPI_DYNAMIC_CONNECTION=0
  export I_MPI_SCALABLE_OPTIMIZATION=0
else
  #. /opt/crtdc/openmpi/1.10.0-hfi-v3/bin/mpivars.sh
  #. /opt/crtdc/openmpi/scripts/mpivars_help.sh
  #PATH=/opt/crtdc/openmpi/1.10.0-hfi-v3/bin:${PATH}; export PATH
  #export LD_LIBRARY_PATH=/opt/intel/compiler/2016u2/compilers_and_libraries_2016.2.181/linux/compiler/lib/intel64/:$LD_LIBRARY_PATH
  KNLOPAOPTSOPENMPI="-mca mtl psm2 -mca btl ^sm -x LD_LIBRARY_PATH=/opt/crtdc/openmpi/1.10.0-hfi-v3/lib64:/opt/intel/compiler/2016u2/compilers_and_libraries_2016.2.181/linux/compiler/lib/intel64/:$LD_LIBRARY_PATH"
  KNLOPAOPTSOPENMPI+="-x PSM2_MQ_RNDV_HFI_WINDOW=4194304 -x PSM2_MQ_EAGER_SDMA_SZ=65536 -x PSM2_MQ_RNDV_HFI_THRESH=200000"
fi

# Client affinity
maxthreadid=$((numthreads-1))
maxcores=`cpuinfo | grep "Cores per package" | awk '{print $5}'`
affinitystr="proclist=[0-5,$((5+1+1))-$((maxcores-1))],granularity=thread,explicit"

CENV="$ENV1 $ENV2 -env I_MPI_PIN_DOMAIN [10]"
list=5

export KMP_AFFINITY=$affinitystr
export OMP_NUM_THREADS=$numthreads
export KMP_PLACE_THREADS=1t
echo THREAD SETTINGS: Affinity $affinitystr Threads $numthreads Placement $KMP_PLACE_THREADS

if [ "$fabric" != "openmpi" ]; then
  echo "mpiexec.hydra -ppn $ppncpu -np $numnodes -hostfile $hostfile $CENV $CPRO numactl -m 1 $bin"
  mpiexec.hydra -ppn $ppncpu -np $numnodes -hostfile $hostfile $CENV $CPRO numactl -m 1 ./$bin
else
  echo "mpirun -npernode $ppncpu -np $numnodes -hostfile $hostfile $KNLOPAOPTSOPENMPI taskset -c $list numactl -m 1 $bin"
  mpirun -npernode $ppncpu -np $numnodes -hostfile $hostfile $KNLOPAOPTSOPENMPI -x KMP_AFFINITY="$affinitystr" -x OMP_NUM_THREADS=$numthreads -x KMP_PLACE_THREADS=1t taskset -c $list numactl -m 1 ./$bin
fi

}


if [ $# -ne 2 ]
then
  echo "Usage: run_allreduce_ia.sh <osu_allreduce binary> <hostfile>"
  exit
fi

bin=$1
hostfile=$2
outdir=

#build with Intel MPI - expect Intel MPI (mpiicc) to be in path
mpiicc -g -Wall -O3 -std=gnu99  -o osu_allreduce *.c

nodelist="2 4 8 16 32"

for nodes in $nodelist; do
  echo "run_osu $nodes $bin 1 $hostfile > ./$outdir/out-$nodes.txt"
  run_osu $nodes $bin 1 $hostfile > ./$outdir/out-$nodes.txt
done

#******************************************************************************
# Copyright 2017 Intel Corporation
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

#!/bin/sh

CUR_DIR=$(dirname `which $0`)

function usage
{
    echo "usage: run_allreduce_ia.sh [hostfile] [allreduce_binary]"
    echo "         hostfile: file with one hostname per line"
    echo "         allreduce_binary: name of binary with allreduce benchmark"
    echo "                           possible values: osu_allreduce"
    echo "                                            mlsl_osu_allreduce"
}

function run_allreduce
{
    proc_count=$1
    ppn=$2
    host_file=$3
    numa_node=$4
    binary=$5

    # IMPI configuration
    export I_MPI_FABRICS=tmi
    export I_MPI_PIN_PROCESSOR_LIST=5
    export I_MPI_DEBUG=5

    # MLSL configuration
    export MLSL_NUM_SERVERS=16
    export MLSL_SERVER_AFFINITY="6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21"
    export MLSL_LOG_LEVEL=1

    mpiexec.hydra -bootstrap ssh -n $proc_count -ppn $ppn -hostfile $hostfile numactl --preferred=$numa_node $binary
}

function cleanup_nodes
{
    node_count=$1
    hostfile=$2

    node_idx=1
    for node in `cat $hostfile`;
    do
        echo "cleanup node $node, idx $node_idx"
        ssh -n $node "killall -q mpirun mpiexec.hydra pmi_proxy ep_server; rm -rf /dev/shm/*"
        if [ ${node_idx} == ${node_count} ]; then
            break
        fi
        node_idx=$((node_idx+1))
    done
}


if [ $# -ne 2 ]
then
  usage
  exit
fi

hostfile="${CUR_DIR}/$1"
binary_name=$2
binary_path="${CUR_DIR}/bin/$binary_name"
output_dir="${CUR_DIR}/out"

proc_counts="2 4 8 16 32"
binary_names="osu_allreduce mlsl_osu_allreduce"
ppn=1
iter_count=10

binary_found=0
for bname in $binary_names
do
    if [ "$bname" == "$binary_name" ]; then
        binary_found=1
        break
    fi
done

if [ "$binary_found" == "0" ]; then
    echo "unexpected binary \"$binary_name\""
    usage
    exit
fi


if [ ! -f "$binary_path" ]; then
    echo "build binary \"$binary_name\" at first, execute \"make $binary_name\""
    usage
    exit
fi

if [ ! -f "$hostfile" ]; then
    echo "create hostfile at first"
    usage
    exit
fi

mkdir -p $output_dir
host_count=(`cat ${hostfile} | grep -v ^$ | wc -l`)

# KNL specific
numa_node_count=($(numactl -H | grep "available" | awk -F ' ' '{print $2}'))
if [ $numa_node_count -eq 1 ]; then
    echo "MCDRAM is in cache mode"
    numa_node=0
else
    echo "MCDRAM is in flat mode"
    numa_node=1
fi

for proc_count in $proc_counts
do
    if [ "$host_count" -lt "$proc_count" ]; then
        echo "not enough hosts for process count $proc_count"
        continue
    fi

    cleanup_nodes $proc_count $hostfile

    for iter in `seq $iter_count`
    do
        echo "parameters: proc_count $proc_count, binary $binary_name, iter $iter"
        run_allreduce $proc_count $ppn $hostfile $numa_node $binary_path | tee "$output_dir/$binary_name-$proc_count-$iter.txt"
    done
done

cleanup_nodes $host_count $hostfile

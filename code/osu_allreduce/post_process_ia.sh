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

#!/bin/bash

CUR_DIR=$(dirname `which $0`)

function usage
{
    echo "The script retrieves 'median' and 'standard deviation' metrics "
    echo "from allreduce log files gathered on multiple launches."
    echo "usage: post_process.sh [allreduce_binary]"
    echo "         allreduce_binary: name of binary with allreduce benchmark"
    echo "                           possible values: osu_allreduce"
    echo "                                            mlsl_osu_allreduce"
}

binary_name=$1
msg_sizes=`cat "../kernels/all_reduce_problems.h" | grep "all_reduce_kernels_size" | awk '{print substr($0, index($0, $4))}' | tr -d ',;{}'`
proc_counts="2 4 8 16 32"

binary_names="osu_allreduce mlsl_osu_allreduce"
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

output_file="${CUR_DIR}/$binary_name.txt"

rm -f $output_file

for msg_size in $msg_sizes
do
    echo "size (# of floats): $msg_size" >> $output_file
    for proc_count in $proc_counts
    do
        if test -z "$(ls -l ${CUR_DIR}/out | grep $binary_name-$proc_count | head -n 1)"
        then
            continue
        fi
        AVG_MSEC=`cat ${CUR_DIR}/out/$binary_name-$proc_count-* | grep "$msg_size  " | awk '{ sum += $2 } END { print sum / NR }'`
        STD_DEV_MSEC=`cat ${CUR_DIR}/out/$binary_name-$proc_count-* | grep "$msg_size  " | awk -vAVG=$AVG_MSEC '{ sum += ($2-AVG) * ($2-AVG); } END { print sqrt(sum / NR) }'`
        AVG_MSEC=`printf '%.2f\n' $(echo "$AVG_MSEC")`
        STD_DEV_MSEC=`printf '%.2f\n' $(echo "$STD_DEV_MSEC")`
        echo "$AVG_MSEC $STD_DEV_MSEC" >> $output_file
    done
done

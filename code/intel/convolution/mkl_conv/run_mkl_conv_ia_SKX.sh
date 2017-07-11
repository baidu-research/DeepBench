#!/bin/bash

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

echo "------------------------------------------------------------"
echo " MKL Convolution - SKX (training, float32)"
echo "------------------------------------------------------------"
echo " "
./std_conv_bench --training --f32

echo "------------------------------------------------------------"
echo " MKL Convolution - SKX (inference-server, float32)"
echo "------------------------------------------------------------"
echo " "
./std_conv_bench --inference-server --f32

echo "------------------------------------------------------------"
echo " MKL Convolution - SKX (inference-server, int8)"
echo "------------------------------------------------------------"
echo " "
./std_conv_bench --inference-server --int8

echo "------------------------------------------------------------"
echo " MKL Convolution - SKX (inference-device, float32)"
echo "------------------------------------------------------------"
echo " "
./std_conv_bench --inference-device --f32

echo "------------------------------------------------------------"
echo " MKL Convolution - SKX (inference-device, int8)"
echo "------------------------------------------------------------"
echo " "
./std_conv_bench --inference-device --int8


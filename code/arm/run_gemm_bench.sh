#!/bin/bash

git submodule init && git submodule update
rm -rf ./bin
make gemm
echo build success!
echo start running!
bin/gemm_bench
echo running complete!

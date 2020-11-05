#!/bin/bash

source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh

spack load python@3.8.0
spack load dawn4py
spack load dawn%gcc

alias clang-format="/project/c16/easybuild/software/clang+llvm/7.0.0-x86_64-linux-sles12.3/bin/clang-format"

spack build-env --dump build.env atlas_utilities%gcc >/dev/null
source build.env
rm build.env

spack load atlas@0.22.0
spack load atlas_utilities@master
spack load cuda@10.1.243
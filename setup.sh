#!/bin/bash

source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh

# TODO: test from scratch!!
spack install --overwrite dawn4py
spack install --overwrite py-pip%gcc ^python@3.8.0
spack install --overwrite dawn%gcc
spack install --overwrite atlas%gcc
spack install --overwrite atlas_utilities%gcc

spack load python@3.8.0
spack load dawn4py
spack load py-pip%gcc ^python@3.8.0

pip install dusk@git+https://github.com/dawn-ico/dusk.git

dusk --help
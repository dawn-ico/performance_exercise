#!/bin/bash

spack install -y --overwrite python@3.8.0
spack install -y --overwrite dawn4py
spack install -y --overwrite py-pip%gcc ^python@3.8.0
spack install -y --overwrite dawn%gcc
spack install -y --overwrite atlas%gcc
spack install -y --overwrite atlas_utilities%gcc

spack load python@3.8.0
spack load dawn4py
spack load py-pip%gcc ^python@3.8.0

pip install dusk@git+https://github.com/dawn-ico/dusk.git

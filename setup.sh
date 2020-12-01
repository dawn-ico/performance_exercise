#!/bin/bash

source ~/spack/share/spack/setup-env.sh

spack load python@3.8.0
spack load dawn4py
spack load py-pip%gcc ^python@3.8.0

pip install dusk@git+https://github.com/dawn-ico/dusk.git

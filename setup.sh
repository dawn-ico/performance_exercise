#!/bin/bash

spack install -y python@3.8.0
spack install -y dawn4py
spack install -y py-pip%gcc ^python@3.8.0
spack install -y dawn%gcc
spack install -y atlas%gcc
spack install -y atlas_utilities%gcc ^netcdf-c -mpi ^hdf5 -mpi

spack load python@3.8.0
spack load dawn4py
spack load py-pip%gcc ^python@3.8.0

pip install dusk@git+https://github.com/dawn-ico/dusk.git

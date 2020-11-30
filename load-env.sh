#!/bin/bash

spack load python@3.8.0
spack load dawn4py
spack load dawn%gcc

hash=$(spack find --format "{hash}" atlas_utilities%gcc ^netcdf-c -mpi ^hdf5)

spack build-env --dump build.env /${hash} >/dev/null
source build.env
rm build.env

spack load atlas@0.22.0
spack load /${hash}
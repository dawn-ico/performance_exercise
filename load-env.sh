#!/bin/bash

spack load python@3.8.0
spack load dawn4py
spack load dawn%gcc

spack build-env --dump build.env atlas_utilities%gcc >/dev/null
source build.env
rm build.env

spack load atlas@0.22.0
spack load atlas_utilities@master
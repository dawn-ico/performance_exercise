#!/bin/bash

# NEED TO source load-env.sh FIRST!
echo "nvcc compile"
nvcc --compile -I./ -o exercise_cuda.o --ptxas-options=-v exercise_cuda.cu
echo "g++ compile driver"
g++ -c -std=c++17 -I/usr/local/cuda/include/ -I./ exercise_driver.cpp utils/atlasToGlobalGpuTriMesh.cpp
echo "g++ link"
g++ exercise_cuda.o exercise_driver.o atlasToGlobalGpuTriMesh.o -o exercise -L/usr/local/cuda/lib64/ -lcuda -lcudart -latlasIOLib -latlasUtilsLib -latlas -leckit
chmod +x exercise
rm -rf *.o
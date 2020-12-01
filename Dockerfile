# Use the official image as a parent image.
FROM nvidia/cuda:11.1-devel-ubuntu20.04

# Set the working directory.
WORKDIR /root

RUN apt-get update && apt-get -y install gcc-8 g++-8 build-essential curl git clang-format vim unzip python3-pip gfortran-8

RUN pip3 install pyyaml

RUN git clone https://github.com/spack/spack ~/spack
RUN git config --global url."https://github.com/".insteadOf git@github.com:

RUN git clone https://github.com/MeteoSwiss-APN/spack-mch.git && git clone -b homework https://github.com/dawn-ico/performance_exercise.git
RUN rm -rf spack-mch/packages/cosmo*
RUN . ~/spack/share/spack/setup-env.sh && spack repo add spack-mch/ && spack compiler find
RUN . ~/spack/share/spack/setup-env.sh && spack compiler remove gcc@9; exit 0
RUN . ~/spack/share/spack/setup-env.sh && spack compiler remove gcc@10; exit 0

WORKDIR /root/performance_exercise

RUN . ~/spack/share/spack/setup-env.sh && spack install -y python@3.8.0
RUN . ~/spack/share/spack/setup-env.sh && spack install -y dawn4py
RUN . ~/spack/share/spack/setup-env.sh && spack install -y py-pip%gcc ^python@3.8.0
RUN . ~/spack/share/spack/setup-env.sh && spack install -y dawn%gcc
RUN . ~/spack/share/spack/setup-env.sh && spack install -y atlas%gcc
RUN . ~/spack/share/spack/setup-env.sh && spack install -y atlas_utilities%gcc ^netcdf-c -mpi ^hdf5 -mpi
RUN git pull
RUN ./setup.sh
RUN /bin/bash -c "source ./load-env.sh && ./compile.sh"

CMD ["/bin/bash"]
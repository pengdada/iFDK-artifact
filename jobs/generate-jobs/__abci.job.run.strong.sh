#!/usr/bin/env bash
module load intel-mpi/2018.2.199

export LOCAL_HOME=$HOME/local
export LD_LIBRARY_PATH=$LOCAL_HOME/lib
export IPP_HOME=$LOCAL_HOME/intel/compilers_and_libraries_2018.1.163/linux/ipp
export LD_LIBRARY_PATH=$LOCAL_HOME/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:$LD_LIBRARY_PATH
export MPIROOT=/apps/intel/2018.2/compilers_and_libraries_2018.2.199/linux/mpi/intel64
export PATH=$MPIROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPIROOT/lib:$LD_LIBRARY_PATH
export MANPATH=$MANPATH:$MPIROOT/share/man

echo PATH=$PATH

LD_PATH=$HOME/local/cuda-9.0/lib64:$MPIROOT/lib64:$(echo $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$LD_PATH

echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo LOCAL_PATH=$LOCAL_HOME
echo HOME=$HOME

MPIRUN=$(which mpiexec)
mode=Release
process_name=iFDK

echo MPIRUN=$MPIRUN
echo process_name=$process_name
#############################


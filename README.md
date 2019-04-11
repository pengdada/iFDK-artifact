# iFDK: A Scalable Framework for Instant High-resolution Image Reconstruction
We use ABCI (https://abci.ai/) HPC to solve the high-resolution image reconstruction problem, e.g. 2048^3, 4096^3, 8192^3.
This repository contains the execution modules, job scripts on HPC and benchmarks.

## DEPENDENCIES
We tested on ABCI, using Nvidia Volta V100 GPUs, with GCC 4.8 and NVCC 9.0. 
The following libraries and tools are requirements:

    cmake >= 3.1
    CUDA >=9.0
    INTEL MPI 2018.2.199
    INTEL IPP 2018.2.199
    
## BUILD/RUN-TIME ENVIRONMENT
We make a folder, called "local" at user's home directory. All tools, e.g. GCC, LLVM, CUDA-10.0, are managed in this folder as.
   

## DATA


## BENCHMARKS
The related benchmarks can be found in follows:
https://www.openrtk.org/

https://github.com/LLNL/ior

https://docs.nvidia.com/cuda/cuda-samples/index.html
    

## MORE INFORMATION
For more information or questions, contact the authors at chinhou0718#gmail.com (replace # by @, please)


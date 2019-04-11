# iFDK: A Scalable Framework for Instant High-resolution Image Reconstruction
We use HPC to solve the high-resolution image reconstruction problem, e.g. 2048^3, 4096^3, 8192^3.
This repository contains the execution modules, job scripts on HPC and benchmarks.

## DEPENDENCIES
We tested on ABCI, using Nvidia Volta V100 GPUs, with GCC 4.8 and NVCC 9.0. 
The following libraries and tools are requirements:

    cmake >= 3.1
    GCC >=5.4
    NVCC >=10.0
    LLVM >=7.0
    FFTW >=3.3.8
    PPCG >= 0.08
    Halide >= 2018/02/15
    cuDNN >= 7.4
    
## BUILD/RUN-TIME ENVIRONMENT
We make a folder, called "local" at user's home directory. All tools, e.g. GCC, LLVM, CUDA-10.0, are managed in this folder as.
   
        ~/local/
        ├── cmake-3.13.1
        ├── ctages-5.8
        ├── cuda-10.0
        ├── fftw-3.3.8
        ├── gcc-5.4
        ├── gdb-8.2
        ├── Halide
        ├── ppcg
        ├── llvm-7.0.1


## DATA


## 2D CONVOLUTION




## 2D/3D STENCIL

- SSAM : src/ssai-2dconv,  src/ssai-3dstencil,  src/ssai-j3d27pt,  src/ssai-poisson,
- ppcg : src/ppcg-j2d121pt,  src/ppcg-j2d17pt,  src/ppcg-j2d25pt,  src/ppcg-j2d64pt,  src/ppcg-j2ds25pt,  src/ppcg-j3d27pt,  src/ppcg-poisson,
- Halid : src/halid-2dstencil, src/halid-3dstencil

- How to run

	go to the SSAM root directory and all of the 2D/3D stencil result can be abtained by running
		
	./test-stencil.sh
	

## BENCHMARKS
The related benchmarks can be found in follows:

    

## MORE INFORMATION
For more information or questions, contact the authors at chinhou0718#gmail.com (replace # by @, please)


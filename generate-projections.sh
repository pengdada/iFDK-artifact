#!/bin/bash

echo ""--------------------------------------------------
DST=$HOME/gen-data/
mkdir -p $DST
SRC=./data/Shepp-Logan-512x512x512.vol


LD=$LOCAL_HOME/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD

#!/bin/bash
#for w in 512 1024 2048 4096 8192
for w in 2048
do
	for n in 32768
	do
	    ./Release/genProj $DST $SRC $w $w $n
        done
done

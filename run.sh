#!/bin/bash

# for scaling evaluation on abci hpc

# repeate N times

N=1

echo "#########################"
echo "each job repeat $N times"
jobs="strong2k strong4k strong8k weak2k weak4k weak8k"

function run_job(){
	pushd jobs/generate-jobs
		./clear.sh
		python gen-jobs.py $1
	popd
}


echo $#

for s in strong2k strong4k strong8k weak2k weak4k weak8k; do	
	if [ $# -eq 1 ]; then
		if [ "$s" == "$1" ]; then
			echo "run all jobs : $s"
		else
			continue
		fi
	fi
	run_job $s
	batch-sub-jobs.sh $N

	echo "$s result is in folder test"
	mkdir -p test
	mkdir -p test/$s
	mv *.t1 test/$s
	mv *.t2 test/$s
	mv abci.job.run.* test$s
done





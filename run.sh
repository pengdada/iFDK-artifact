#!/bin/bash

# repeate N times

N=1
echo "each job repeat $N times"

for s in strong2k strong4k strong8k weak2k weak4k weak8k; do
	pushd jobs/generate-jobs
		./clear.sh
		python gen-jobs.py $s
	popd
	batch-sub-jobs.sh $N
	mkdir -p test
	mkdir -p test/$s
	mv *.t1 test/$s
	mv *.t2 test/$s
	mv abci.job.run.* test$s
done





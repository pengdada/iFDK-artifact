#!/bin/sh



for s in strong2k strong4k strong8k weak2k weak4k weak8k; do
	./clear.sh
	python gen-jobs.py $s
	mkdir -p ../$s
	rm -rf ../$s/*
	mv abci.job.run.* ../$s
done

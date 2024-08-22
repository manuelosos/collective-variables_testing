#!/bin/bash

target="bt705242@emil.rz.uni-bayreuth.de:/scratch/bt705242/collective-variables_testing/runfiles"

rsync -avz test_space/tests/cluster_runfiles/ ${target}
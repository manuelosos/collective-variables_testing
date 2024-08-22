#!/bin/bash

source="bt705242@emil.rz.uni-bayreuth.de:/scratch/bt705242/collective-variables_testing/results/"

rsync -avz ${source} /home/manuel/Documents/Studium/praktikum/code/sponet_cv_testing/test_space/tmp_results/

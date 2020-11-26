#!/bin/bash

# Run script but detach process and capture output in a log file

cd cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

nohup python python_src/TestHybridGreedy.py > log-pso.txt &
#!/bin/bash

# Run script but detach process and capture output in a log file

cd cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

git checkout instance/uber

workon wesselsenv

nohup python python_src/TestPsoMain.py > log-pso.txt &

nohup python python_src/TestControlMain.py > log-control.txt &
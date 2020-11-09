#!/bin/bash

# Run script but detach process and capture output in a log file

cd masters-tensorflow2-model/

workon wesselsenv

nohup python python_src/TestPsoMain.py > log-pso.txt &

nohup python python_src/TestControlMain.py > log-control.txt &
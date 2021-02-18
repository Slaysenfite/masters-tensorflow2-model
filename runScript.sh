#!/bin/bash

# Run script but detach process and capture output in a log file

cd cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# No metaheuristic (sgd)
nohup python python_src/ResNetMain.py > log-pso.txt &

# Pso First (sgd)
#nohup python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=first > log-pso.txt &

# Pso Last (sgd)
#nohup python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=last > log-pso.txt &

# Ga First (sgd)
#nohup python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=first > log-pso.txt &

# Ga First (sgd)
#nohup python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=last > log-pso.txt &
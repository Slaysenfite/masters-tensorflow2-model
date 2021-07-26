#!/bin/bash

# Run script but detach process and capture output in a log file

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

python python_src/UNetMain.py --optimizer=sgd --id=MCUD001 --dataset=cbis
python python_src/UNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCUD002 --dataset=cbis
python python_src/UNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCUD003 --dataset=cbis
python python_src/UNetMain.py --optimizer=sgd --id=MCUD001A --augmentation=True --dataset=cbis
python python_src/UNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCUD002A --augmentation=True --dataset=cbis
python python_src/UNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCUD003A --augmentation=True --dataset=cbis

#python python_src/UNetMain.py --optimizer=sgd --id=MCUB001 --dataset=bsc
#python python_src/UNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCUB002 --dataset=bsc
#python python_src/UNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCUB003 --dataset=bsc
#python python_src/UNetMain.py --optimizer=sgd --id=MCUB001A --augmentation=True --dataset=bsc
#python python_src/UNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCUB002A --augmentation=True --dataset=bsc
#python python_src/UNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCUB003A --augmentation=True --dataset=bsc

python python_src/ResNetMain.py --optimizer=sgd --id=MCRD001 --dataset=cbis
python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCRD002 --dataset=cbis
python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCRD003 --dataset=cbis
python python_src/ResNetMain.py --optimizer=sgd --id=MCRD001A --augmentation=True --dataset=cbis
python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCRD002A --augmentation=True --dataset=cbis
python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCRD003A --augmentation=True --dataset=cbis

#python python_src/ResNetMain.py --optimizer=sgd --id=MCRB001 --dataset=bsc
#python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCRB002 --dataset=bsc
#python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCRB003 --dataset=bsc
#python python_src/ResNetMain.py --optimizer=sgd --id=MCRB001A --augmentation=True --dataset=bsc
#python python_src/ResNetMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MCRB002A --augmentation=True --dataset=bsc
#python python_src/ResNetMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MCRB003A --augmentation=True --dataset=bsc

python python_src/UNetSegMain.py --optimizer=sgd --id=MSUD001
python python_src/UNetSegMain.py --meta_heuristic=pso --meta_heuristic_order=last  --id=MSUD002
python python_src/UNetSegMain.py --meta_heuristic=ga --meta_heuristic_order=last  --id=MSUD003

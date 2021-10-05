#!/bin/bash

# Run script but detach process and capture output in a log file

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

## CONTROLS ##

#python python_src/ResNetMain.py --optimizer=adam --id=MCRD001 --dataset=cbis --preloaded_weights=False --epochs=100
#python python_src/ResNetMain.py --optimizer=adam --id=MCRD001-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
#
#python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001 --dataset=cbis --preloaded_weights=False --epochs=100
#python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
#
#python python_src/UNetSegMain.py --id=MSUD001 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=100
#python python_src/UNetSegMain.py --id=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=100

python python_src/ResNetMain.py --optimizer=adam --id=MCRF001-ImageNet --dataset=ddsm --preloaded_weights=True --epochs=100
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXF001-ImageNet --dataset=ddsm --preloaded_weights=True --epochs=100

## Metaheuristics ##

python python_src/NewResNetMain.py --id=CRF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga
python python_src/UNetSegMainNew.py --id=SUF2-GA --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --epochs=100 --meta_heuristic=ga

python python_src/NewResNetMain.py --id=CRF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso
python python_src/NewXceptionNetMain.py --id=CXF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso
python python_src/UNetSegMainNew.py --id=SUF2-PSO --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --epochs=100 --meta_heuristic=pso


python python_src/NewResNetMain.py --id=CRF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga --dataset=ddsm
python python_src/NewXceptionNetMain.py --id=CXF2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga --dataset=ddsm

python python_src/NewResNetMain.py --id=CRF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso --dataset=ddsm
python python_src/NewXceptionNetMain.py --id=CXF2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso --dataset=ddsm
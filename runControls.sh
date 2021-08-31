#!/bin/bash

Ccd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A --dataset=cbis --preloaded_weights=True --augmentation=True --epochs=100
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001 --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001b --dataset=cbis --preloaded_weights=True --epochs=100 --l2=0.00008

python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001A --dataset=cbis --preloaded_weights=True --augmentation=True --epochs=100
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001 --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001b --dataset=cbis --preloaded_weights=True --epochs=100 -l2=0.00008

#python python_src/UNetMain.py --optimizer=adam --id=MCUD001A --dataset=cbis --preloaded_weights=True --augmentation=True --epochs=100
#python python_src/UNetMain.py --optimizer=adam --id=MCUD001 --dataset=cbis --preloaded_weights=True --epochs=100

python python_src/UNetSegMain.py --id=MSUD001A --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --augmentation=True --epochs=100
python python_src/UNetSegMain.py --id=MSUD001 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=100

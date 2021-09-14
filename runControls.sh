#!/bin/bash

Ccd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

python python_src/ResNetMain.py --optimizer=adam --id=MCRD001 --dataset=cbis --preloaded_weights=False --epochs=100
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001-Z --dataset=cbis --preloaded_weights=True --epochs=100 --l2=0.8 --kernel_initializer=zeros

python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001 --dataset=cbis --preloaded_weights=False --epochs=100
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXD001-Z --dataset=cbis --preloaded_weights=True --epochs=100 --l2=0.012 --kernel_initializer=zeros

python python_src/UNetSegMain.py --id=MSUD001 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=100
python python_src/UNetSegMain.py --id=MSUD001-ImageNet --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=100
python python_src/UNetSegMain.py --id=MSUD001A --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --augmentation=True --epochs=100

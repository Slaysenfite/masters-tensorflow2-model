#!/bin/bash

Ccd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

#python python_src/UNetMain.py --optimizer=adam --id=MCUD001A01 --augmentation=True --dataset=cbis --tf_fit=False --preloaded_weights=False
#python python_src/UNetMain.py --optimizer=adam --id=MCUD001A02 --augmentation=True --dataset=cbis --tf_fit=False --preloaded_weights=True
python python_src/UNetMain.py --optimizer=adam --id=MCUD001A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True
python python_src/UNetMain.py --optimizer=adam --id=MCUD00103 --augmentation=False --dataset=cbis --tf_fit=True --preloaded_weights=True
#python python_src/UNetMain.py --optimizer=adam --id=MCUD001A04 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=False


#python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A01 --augmentation=True --dataset=cbis --tf_fit=False --preloaded_weights=False
#python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A02 --augmentation=True --dataset=cbis --tf_fit=False --preloaded_weights=True
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True
python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True
#python python_src/ResNetMain.py --optimizer=adam --id=MCRD001A04 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=False

#python python_src/UNetSegMain.py --id=MSUD00101 --optimizer=sgd
#python python_src/UNetSegMain.py --id=MSUD00102 --optimizer=sgd --tf_fit=False --preloaded_weights=True
python python_src/UNetSegMain.py --id=MSUD00103  --optimizer=sgd --tf_fit=True --preloaded_weights=True

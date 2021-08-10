#!/bin/bash

# Run script but detach process and capture output in a log file

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

python python_src/UNetMain.py --optimizer=adam --id=MCUD002A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True --meta_heuristic=pso --meta_heuristic_order=last
python python_src/UNetMain.py --optimizer=adam --id=MCUD003A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True --meta_heuristic=ga --meta_heuristic_order=last

python python_src/ResNetMain.py --optimizer=adam --id=MCRD002A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True --meta_heuristic=pso --meta_heuristic_order=last
python python_src/ResNetMain.py --optimizer=adam --id=MCRD003A03 --augmentation=True --dataset=cbis --tf_fit=True --preloaded_weights=True --meta_heuristic=ga --meta_heuristic_order=last

python python_src/UNetSegMain.py --id=MSUD00203  --optimizer=sgd --tf_fit=True --preloaded_weights=True --meta_heuristic=pso --meta_heuristic_order=last
python python_src/UNetSegMain.py --id=MSUD00303  --optimizer=sgd --tf_fit=True --preloaded_weights=True --meta_heuristic=ga --meta_heuristic_order=last

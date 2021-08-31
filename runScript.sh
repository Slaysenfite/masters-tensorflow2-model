#!/bin/bash

# Run script but detach process and capture output in a log file

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

workon wesselsenv

# Run Scripts

python python_src/ResNetMain.py --optimizer=adam --id=MCRDPso --dataset=cbis --tf_fit=True --meta_heuristic=pso  --preloaded_experiment=MCRD001
python python_src/ResNetMain.py --optimizer=adam --id=MCRDGa --dataset=cbis --tf_fit=True --meta_heuristic=ga  --preloaded_experiment=MCRD001

python python_src/XceptionNetMain.py --optimizer=adam --id=MCXDPso --dataset=cbis --tf_fit=True --meta_heuristic=pso  --preloaded_experiment=MCXD001
python python_src/XceptionNetMain.py --optimizer=adam --id=MCXDGa --dataset=cbis --tf_fit=True --meta_heuristic=ga  --preloaded_experiment=MCXD001X

python python_src/UNetSegMain.py --id=MSUDPso  --optimizer=sgd --tf_fit=True --meta_heuristic=pso  --preloaded_experiment=MSUD001
python python_src/UNetSegMain.py --id=MSUDGa  --optimizer=sgd --tf_fit=True --meta_heuristic=ga  --preloaded_experiment=MSUD001
